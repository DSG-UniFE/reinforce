# frozen_string_literal: true

require "torch"
require_relative "../networks"
require_relative "../hyperparameters"
require_relative "../running_mean_std"

module Reinforce
  module Algorithms
    # Soft Actor-Critic for discrete action spaces.
    class SAC
      include ::Reinforce::Agent

      # Default target entropy, as a fraction of the maximum (uniform-policy)
      # entropy log(|actions|), for automatic entropy tuning. The original
      # discrete-SAC paper used 0.98, which keeps the policy almost uniform:
      # on Taxi, tuning then raised alpha until the entropy bonus outweighed
      # the -1 step penalty, so wandering paid better than finishing and the
      # greedy policy was arbitrary. Measured on GridWorld and Taxi (with the
      # random-action warmup below), 0.98 and 0.89 failed most seeds, 0.6 was
      # inconsistent on a 10x10 GridWorld, and 0.3 solved 22 of 25 seeds.
      TARGET_ENTROPY_SCALE = 0.3

      # Added to the return variance before taking its square root when
      # normalizing rewards, as in Gymnasium's NormalizeReward.
      REWARD_NORMALIZATION_EPSILON = 1e-8

      # Hyperparameters (see Reinforce::Hyperparameters).
      #
      # entropy_coefficient: alpha, or its initial value when
      #   auto_entropy_tuning is on.
      # target_entropy: the entropy automatic tuning aims for; nil means
      #   TARGET_ENTROPY_SCALE * log(number of actions).
      # warmup_steps: initial environment steps (across all #train calls)
      #   that take uniformly random actions and make no updates.
      # tau: Polyak averaging factor for the target critics.
      # normalize_rewards: divide rewards by a running estimate of the
      #   standard deviation of the discounted return (see #reward_scale).
      #   With it, SAC solved 25 of 25 seeds across GridWorld (5x5, 10x10)
      #   and Taxi, against 22 of 25 without.
      Config = Data.define(
        :learning_rate, :discount_factor, :entropy_coefficient, :auto_entropy_tuning, :target_entropy,
        :batch_size, :replay_size, :tau, :warmup_steps,
        :critic_updates_per_step, :policy_updates_per_step, :target_updates_per_step, :normalize_rewards
      ) do
        def initialize(learning_rate: 3e-4, discount_factor: 0.99, entropy_coefficient: 0.1, auto_entropy_tuning: true,
          target_entropy: nil, batch_size: 64, replay_size: 10_000, tau: 0.05, warmup_steps: 2_000,
          critic_updates_per_step: 1, policy_updates_per_step: 1, target_updates_per_step: 1, normalize_rewards: true)
          super
        end
      end

      attr_reader :metrics, :config, :policy_model, :q1_model, :q2_model, :q1_target_model, :q2_target_model, :actions, :batch_size

      alias_method :logs, :metrics

      # @parameter intrinsic_reward [#observe, #call, #train, nil] an
      #   exploration bonus added to sampled rewards (see
      #   Reinforce::IntrinsicRewards).
      def initialize(environment, config = nil, policy_model:, q1_model:, q2_model:, q1_target_model:, q2_target_model:,
        intrinsic_reward: nil, **hyperparameters)
        @config = Reinforce::Hyperparameters.resolve(Config, config, hyperparameters)
        @intrinsic_reward = intrinsic_reward
        learning_rate, discount_factor, entropy_coefficient, auto_entropy_tuning, target_entropy,
          batch_size, replay_size, tau, warmup_steps,
          critic_updates_per_step, policy_updates_per_step, target_updates_per_step = @config.to_h.values_at(
            :learning_rate, :discount_factor, :entropy_coefficient, :auto_entropy_tuning, :target_entropy,
            :batch_size, :replay_size, :tau, :warmup_steps,
            :critic_updates_per_step, :policy_updates_per_step, :target_updates_per_step
          )
        @environment = environment
        @actions = environment.actions
        @action_to_index = {}
        @actions.each_with_index { |action, index| @action_to_index[action] = index }
        @state_size = environment.state_size

        @discount_factor = discount_factor
        @entropy_coefficient = entropy_coefficient
        @batch_size = batch_size
        @replay_size = replay_size
        @tau = tau
        @warmup_steps = warmup_steps
        @critic_updates_per_step = critic_updates_per_step
        @policy_updates_per_step = policy_updates_per_step
        @target_updates_per_step = target_updates_per_step

        @policy_model = policy_model
        @q1_model = q1_model
        @q2_model = q2_model
        @q1_target_model = q1_target_model
        @q2_target_model = q2_target_model
        @q1_target_model.load_state_dict(@q1_model.state_dict)
        @q2_target_model.load_state_dict(@q2_model.state_dict)

        @policy_optimizer = Torch::Optim::Adam.new(@policy_model.parameters, lr: learning_rate)
        @q1_optimizer = Torch::Optim::Adam.new(@q1_model.parameters, lr: learning_rate)
        @q2_optimizer = Torch::Optim::Adam.new(@q2_model.parameters, lr: learning_rate)
        @mse = Torch::NN::MSELoss.new
        @auto_entropy_tuning = auto_entropy_tuning
        if @auto_entropy_tuning
          @target_entropy = target_entropy || (TARGET_ENTROPY_SCALE * Math.log(@actions.size))
          @log_alpha = Torch::NN::Parameter.new(
            Torch.tensor([Math.log(entropy_coefficient)], dtype: :float32)
          )
          @alpha_optimizer = Torch::Optim::Adam.new([@log_alpha], lr: learning_rate)
        else
          @target_entropy = target_entropy
          @log_alpha = nil
          @alpha_optimizer = nil
        end

        @replay = []
        @steps_taken = 0
        @normalize_rewards = @config.normalize_rewards
        @return_rms = Reinforce::RunningMeanStd.new
        @return_statistics_updated = false
        @discounted_return = 0.0
        @metrics = Reinforce::Metrics::Recorder.new
      end

      def predict(state, deterministic: true)
        logits = Torch.no_grad { @policy_model.call(state_tensor(state)) }
        @actions[CategoricalDistribution.new(logits: logits).choose(deterministic: deterministic)]
      end

      # The first `warmup_steps` environment steps (counted across all #train
      # calls) take uniformly random actions and make no updates, as in the
      # reference SAC implementations, so the replay buffer starts with
      # diverse experience rather than whatever the randomly initialized --
      # and typically biased -- policy happens to do.
      def train(episodes:, steps_per_episode:, **_kwargs)
        episodes.times do
          state = @environment.reset
          episode_reward = 0.0
          episode_bonus = 0.0
          episode_length = 0

          steps_per_episode.times do
            action = (@steps_taken < @warmup_steps) ? @actions.sample : select_action(state)
            action_index = @action_to_index.fetch(action)
            next_state, reward, done = @environment.step(action_index)
            store_transition(state, action, reward, next_state, done)
            if @intrinsic_reward
              @intrinsic_reward.observe(next_state)
              bonus = @intrinsic_reward.call([next_state]).first
              episode_bonus += bonus
            end
            track_return(reward + (bonus || 0.0)) if @normalize_rewards

            episode_reward += reward
            episode_length += 1
            state = next_state
            @steps_taken += 1

            if @replay.size >= @batch_size && @steps_taken >= @warmup_steps
              @metrics.record_all(train_step)
            end

            break if done
          end

          @discounted_return = 0.0
          @metrics.record(:episode_reward, episode_reward)
          @metrics.record(:intrinsic_reward, episode_bonus) if @intrinsic_reward
          @metrics.record(:episode_length, episode_length)
        end
      end

      def append_transition(state:, action:, reward:, next_state:, done:)
        store_transition(state, action, reward, next_state, done)
      end

      def replay_size
        @replay.size
      end

      def ready_for_update?(minimum_batch_size = @batch_size)
        @replay.size >= minimum_batch_size
      end

      def sample_batch(batch_size = @batch_size)
        raise ArgumentError, "not enough replay data to sample requested batch size" unless ready_for_update?(batch_size)

        batch_size.times.map { @replay[rand(@replay.size)].dup }
      end

      def save(path)
        Torch.save(
          {
            "policy_model" => @policy_model.state_dict,
            "q1_model" => @q1_model.state_dict,
            "q2_model" => @q2_model.state_dict,
            "q1_target_model" => @q1_target_model.state_dict,
            "q2_target_model" => @q2_target_model.state_dict
          },
          path
        )
      end

      def load(path)
        checkpoint = Torch.load(path)
        @policy_model.load_state_dict(checkpoint["policy_model"])
        @q1_model.load_state_dict(checkpoint["q1_model"])
        @q2_model.load_state_dict(checkpoint["q2_model"])
        @q1_target_model.load_state_dict(checkpoint["q1_target_model"])
        @q2_target_model.load_state_dict(checkpoint["q2_target_model"])
        self
      end

      def train_step(batch: nil, reward_override: nil)
        work_batch = batch || sample_batch(@batch_size)
        @intrinsic_reward&.train(work_batch.map { |transition| transition[:next_state] })
        q1_loss = 0.0
        q2_loss = 0.0
        @critic_updates_per_step.times do
          q1_value, q2_value = update_critics(batch: work_batch, reward_override:)
          q1_loss += q1_value
          q2_loss += q2_value
        end
        q1_loss /= @critic_updates_per_step.to_f
        q2_loss /= @critic_updates_per_step.to_f

        policy_loss = 0.0
        entropy = 0.0
        alpha_loss = 0.0
        @policy_updates_per_step.times do
          policy_value, entropy_value, alpha_value = update_policy_and_alpha(batch: work_batch)
          policy_loss += policy_value
          entropy += entropy_value
          alpha_loss += alpha_value
        end
        policy_loss /= @policy_updates_per_step.to_f
        entropy /= @policy_updates_per_step.to_f
        alpha_loss /= @policy_updates_per_step.to_f

        @target_updates_per_step.times { soft_update_targets }
        {
          q1_loss:,
          q2_loss:,
          policy_loss:,
          alpha_loss:,
          entropy:,
          alpha: alpha,
          loss: q1_loss + q2_loss + policy_loss + alpha_loss
        }
      end

      private

      def state_tensor(state_batch)
        if state_batch.is_a?(Torch::Tensor)
          state_batch
        else
          values = (state_batch.is_a?(Array) && state_batch.first.is_a?(Array)) ? state_batch : [state_batch]
          Torch.tensor(values.map { |state| flatten_state(state) }, dtype: :float32)
        end
      end

      def flatten_state(state)
        state.is_a?(Array) ? state.flatten.map(&:to_f) : [state.to_f]
      end

      def select_action(state)
        logits = Torch.no_grad { @policy_model.call(state_tensor(state)) }
        probs = Torch.softmax(logits, dim: 1).to_a.first
        sample_action(probs)
      end

      def sample_action(probabilities)
        threshold = rand
        cumulative = 0.0
        probabilities.each_with_index do |probability, index|
          cumulative += probability
          return @actions[index] if threshold <= cumulative
        end
        @actions.last
      end

      def store_transition(state, action, reward, next_state, done)
        @replay << {
          state: flatten_state(state),
          action_index: @action_to_index.fetch(action),
          reward: reward.to_f,
          next_state: flatten_state(next_state),
          done: done ? 1.0 : 0.0
        }
        @replay.shift while @replay.size > @replay_size
      end

      def tensors_from_batch(batch)
        states = Torch.tensor(batch.map { |transition| transition[:state] }, dtype: :float32)
        actions = Torch.tensor(batch.map { |transition| action_index_from_transition(transition) }, dtype: :int64)
        rewards = Torch.tensor(batch.map { |transition| transition[:reward] }, dtype: :float32)
        next_states = Torch.tensor(batch.map { |transition| transition[:next_state] }, dtype: :float32)
        dones = Torch.tensor(batch.map { |transition| transition[:done] }, dtype: :float32)
        [states, actions, rewards, next_states, dones]
      end

      def update_critics(batch:, reward_override: nil)
        states, actions, rewards, next_states, dones = tensors_from_batch(batch)
        rewards = reward_override if reward_override
        if @intrinsic_reward && !reward_override
          # The replay buffer stores raw rewards; the bonus is added here,
          # under the current statistics, each time a batch is sampled.
          rewards += Torch.tensor(@intrinsic_reward.call(batch.map { |transition| transition[:next_state] }), dtype: :float32)
        end
        rewards *= reward_scale

        q1 = @q1_model.call(states).gather(1, actions.reshape(-1, 1)).reshape(-1)
        q2 = @q2_model.call(states).gather(1, actions.reshape(-1, 1)).reshape(-1)

        target = nil
        Torch.no_grad do
          next_logits = @policy_model.call(next_states)
          next_log_probs = Torch.log_softmax(next_logits, dim: 1)
          next_probs = Torch.softmax(next_logits, dim: 1)
          target_q1 = @q1_target_model.call(next_states)
          target_q2 = @q2_target_model.call(next_states)
          target_q_min = Torch.min(target_q1, target_q2)
          next_v = (next_probs * (target_q_min - alpha * next_log_probs)).sum(1)
          target = rewards + @discount_factor * (1.0 - dones) * next_v
        end

        q1_loss = @mse.call(q1, target)
        @q1_optimizer.zero_grad
        q1_loss.backward
        @q1_optimizer.step

        q2_loss = @mse.call(q2, target)
        @q2_optimizer.zero_grad
        q2_loss.backward
        @q2_optimizer.step

        [q1_loss.item, q2_loss.item]
      end

      def update_policy_and_alpha(batch:)
        states = Torch.tensor(batch.map { |transition| transition[:state] }, dtype: :float32)
        logits = @policy_model.call(states)
        log_probs = Torch.log_softmax(logits, dim: 1)
        probs = Torch.softmax(logits, dim: 1)

        q1 = Torch.no_grad { @q1_model.call(states) }
        q2 = Torch.no_grad { @q2_model.call(states) }
        q_min = Torch.min(q1, q2)
        entropy = -(probs * log_probs).sum(1).mean

        loss = (probs * (alpha * log_probs - q_min)).sum(1).mean
        @policy_optimizer.zero_grad
        loss.backward
        @policy_optimizer.step

        alpha_loss = 0.0
        if @auto_entropy_tuning
          alpha_objective = (@log_alpha * (entropy.detach - @target_entropy)).mean
          @alpha_optimizer.zero_grad
          alpha_objective.backward
          @alpha_optimizer.step
          alpha_loss = alpha_objective.item
        end

        [loss.item, entropy.item, alpha_loss]
      end

      # Reward normalization (Gymnasium's NormalizeReward, adapted to a
      # replay buffer): #train tracks the running variance of the discounted
      # return as it steps the environment, and every sampled batch's
      # rewards are divided by its *current* square root when computing
      # critic targets. The replay buffer keeps raw rewards, so old
      # transitions are rescaled as the statistics evolve instead of
      # keeping whatever scale was current when they were stored. Rewards
      # are only scaled, never shifted, so their sign and the optimal
      # policy are unchanged. 1.0 until #train has stepped the environment
      # (e.g. when transitions only arrive via #append_transition, as in
      # ExDM), or when normalize_rewards is off.
      def reward_scale
        return 1.0 unless @normalize_rewards && @return_statistics_updated

        1.0 / Math.sqrt(@return_rms.var + REWARD_NORMALIZATION_EPSILON)
      end

      def track_return(reward)
        @discounted_return = @discounted_return * @discount_factor + reward
        @return_rms.update(@discounted_return)
        @return_statistics_updated = true
      end

      def soft_update_targets
        ::Reinforce::Networks.soft_update!(target: @q1_target_model, online: @q1_model, tau: @tau)
        ::Reinforce::Networks.soft_update!(target: @q2_target_model, online: @q2_model, tau: @tau)
      end

      def alpha
        @auto_entropy_tuning ? @log_alpha.exp.item : @entropy_coefficient
      end

      def action_index_from_transition(transition)
        return transition[:action_index] if transition.key?(:action_index)

        @action_to_index.fetch(transition[:action])
      end
    end
  end
end
