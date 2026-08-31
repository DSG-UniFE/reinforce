# frozen_string_literal: true

require "torch"
require_relative "../networks"

module Reinforce
  module Algorithms
    # Soft Actor-Critic for discrete action spaces.
    class SAC
      include ::Reinforce::Agent

      attr_reader :logs, :policy_model, :q1_model, :q2_model, :q1_target_model, :q2_target_model, :actions, :batch_size

      def initialize(environment, policy_model:, q1_model:, q2_model:, q1_target_model:, q2_target_model:,
        learning_rate: 3e-4, discount_factor: 0.99, entropy_coefficient: 0.1, auto_entropy_tuning: true,
        target_entropy: nil, batch_size: 64, replay_size: 10_000, tau: 0.05, warmup_steps: 100,
        critic_updates_per_step: 1, policy_updates_per_step: 1, target_updates_per_step: 1)
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
          @target_entropy = target_entropy || (0.98 * Math.log(@actions.size))
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
        @logs = {
          loss: [],
          policy_loss: [],
          q1_loss: [],
          q2_loss: [],
          alpha_loss: [],
          entropy: [],
          alpha: [],
          episode_reward: [],
          episode_length: []
        }
      end

      def predict(state)
        logits = Torch.no_grad { @policy_model.call(state_tensor(state)) }
        action_index = logits.argmax(1).item
        @actions[action_index]
      end

      def train(num_episodes, max_steps_per_episode)
        global_step = 0

        num_episodes.times do
          state = @environment.reset
          episode_reward = 0.0
          episode_length = 0

          max_steps_per_episode.times do
            action = select_action(state)
            action_index = @action_to_index.fetch(action)
            next_state, reward, done = @environment.step(action_index)
            store_transition(state, action, reward, next_state, done)

            episode_reward += reward
            episode_length += 1
            state = next_state
            global_step += 1

            if @replay.size >= @batch_size && global_step >= @warmup_steps
              metrics = train_step
              @logs[:q1_loss] << metrics[:q1_loss]
              @logs[:q2_loss] << metrics[:q2_loss]
              @logs[:policy_loss] << metrics[:policy_loss]
              @logs[:alpha_loss] << metrics[:alpha_loss]
              @logs[:entropy] << metrics[:entropy]
              @logs[:alpha] << metrics[:alpha]
              @logs[:loss] << metrics[:loss]
            end

            break if done
          end

          @logs[:episode_reward] << episode_reward
          @logs[:episode_length] << episode_length
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
