# frozen_string_literal: true

require "torch"
require_relative "../agent"
require_relative "../networks"
require_relative "../hyperparameters"
require_relative "../metrics"
require_relative "../spaces"

module Reinforce
  module Algorithms
    # Soft Actor-Critic for continuous action spaces (Haarnoja et al., 2018),
    # with a squashed Gaussian policy: the actor outputs the mean and log
    # standard deviation of a diagonal Gaussian, and actions are the tanh of
    # its samples, so they lie in [-1, 1] on each component. The environment
    # declares its space with #action_space (a Reinforce::Spaces::Box), and
    # actions are scaled onto it before each step.
    #
    # Optionally, a shield (see Reinforce::Shields) corrects each action
    # before it is executed and adds a safety loss to the actor's:
    #
    # - `shield.call(state, action)` receives the agent's action in
    #   [-1, 1] (the *base* action) and returns the action to execute, also
    #   in [-1, 1]. The replay buffer keeps the base action, with the reward
    #   and next state of the executed one, so the critics learn the value
    #   of proposing an action under the shield. The random warmup actions
    #   are shielded too.
    # - `shield.safety_loss(states, actions)` returns a differentiable scalar
    #   loss for a batch of states and the actor's deterministic actions
    #   (tanh of the mean); the actor minimizes it with weight
    #   `safety_weight`.
    # - `shield.last_info` (optional) returns a Hash of numbers about the
    #   last #call, summed over each episode and recorded as
    #   `shield_<key>` (e.g. `shield_corrections`).
    #
    # Environments can also report numbers per step in `info[:metrics]`
    # (e.g. `{overflow: 2}`), which are summed over each episode and
    # recorded as `episode_<key>`. An `info[:truncated]` step ends the
    # episode without being treated as terminal, so the critics keep
    # bootstrapping across time limits.
    class ContinuousSAC
      include ::Reinforce::Agent

      # Bounds of the policy's log standard deviation, as in the reference
      # implementations.
      LOG_STD_MIN = -20.0
      LOG_STD_MAX = 2.0

      # Hyperparameters (see Reinforce::Hyperparameters).
      #
      # entropy_coefficient: alpha, or its initial value when
      #   auto_entropy_tuning is on.
      # target_entropy: the entropy automatic tuning aims for; nil means
      #   -(number of action components), the usual heuristic.
      # warmup_steps: initial environment steps (across all #train calls)
      #   that take uniformly random actions and make no updates.
      # tau: Polyak averaging factor for the target critics.
      # hidden_size, hidden_layers: the shape of the default networks (see
      #   Reinforce::Networks.mlp), when they are not given.
      # safety_weight: the weight of the shield's safety loss in the actor's
      #   loss; unused without a shield.
      Config = Data.define(
        :learning_rate, :discount_factor, :entropy_coefficient, :auto_entropy_tuning, :target_entropy,
        :batch_size, :replay_size, :tau, :warmup_steps, :hidden_size, :hidden_layers, :safety_weight
      ) do
        def initialize(learning_rate: 3e-4, discount_factor: 0.99, entropy_coefficient: 1.0, auto_entropy_tuning: true,
          target_entropy: nil, batch_size: 256, replay_size: 100_000, tau: 0.005, warmup_steps: 1_000,
          hidden_size: 256, hidden_layers: 2, safety_weight: 0.1)
          super
        end
      end

      attr_reader :metrics, :config, :action_space, :policy_model, :q1_model, :q2_model, :q1_target_model, :q2_target_model, :shield

      alias_method :logs, :metrics

      # @parameter environment [Reinforce::Environment] an environment with
      #   an #action_space returning a Reinforce::Spaces::Box.
      # @parameter policy_model [Torch::NN::Module, nil] maps a batch of
      #   states to 2 * (number of action components) outputs: the means,
      #   then the log standard deviations. Defaults to an MLP.
      # @parameter q1_model, q2_model [Torch::NN::Module, nil] map a batch of
      #   states concatenated with actions to one value each. Default to MLPs.
      # @parameter q1_target_model, q2_target_model [Torch::NN::Module, nil]
      #   networks shaped like the critics, which start as their copies.
      #   Required with custom critics; default to MLPs otherwise.
      # @parameter shield [#call, #safety_loss, nil] see the class comment.
      def initialize(environment, config = nil, policy_model: nil, q1_model: nil, q2_model: nil,
        q1_target_model: nil, q2_target_model: nil, shield: nil, **hyperparameters)
        @config = Reinforce::Hyperparameters.resolve(Config, config, hyperparameters)
        @environment = environment
        @action_space = environment.action_space
        raise ArgumentError, "ContinuousSAC needs a Reinforce::Spaces::Box action space" unless @action_space.is_a?(Reinforce::Spaces::Box)

        @state_size = environment.state_size
        @action_size = @action_space.size
        @shield = shield

        @policy_model = policy_model || build_network(@state_size, 2 * @action_size)
        @q1_model = q1_model || build_network(@state_size + @action_size, 1)
        @q2_model = q2_model || build_network(@state_size + @action_size, 1)
        @q1_target_model = target_network(@q1_model, q1_target_model, custom: !q1_model.nil?)
        @q2_target_model = target_network(@q2_model, q2_target_model, custom: !q2_model.nil?)

        learning_rate = @config.learning_rate
        @policy_optimizer = Torch::Optim::Adam.new(@policy_model.parameters, lr: learning_rate)
        @q_optimizer = Torch::Optim::Adam.new(@q1_model.parameters + @q2_model.parameters, lr: learning_rate)
        if @config.auto_entropy_tuning
          @target_entropy = @config.target_entropy || -@action_size.to_f
          @log_alpha = Torch::NN::Parameter.new(Torch.tensor([Math.log(@config.entropy_coefficient)], dtype: :float32))
          @alpha_optimizer = Torch::Optim::Adam.new([@log_alpha], lr: learning_rate)
        end

        @replay = ReplayBuffer.new(@config.replay_size, @state_size, @action_size)
        @steps_taken = 0
        @metrics = Reinforce::Metrics::Recorder.new
      end

      # @parameter state [Array(Numeric)] a state of the environment.
      # @parameter deterministic [Boolean] tanh of the policy's mean if true,
      #   a sample otherwise.
      # @parameter shielded [Boolean] with a shield, whether it corrects the
      #   action (false shows what the policy proposes by itself).
      # @returns [Array(Float)] an element of the action space, shielded if
      #   the agent has a shield and `shielded`.
      def predict(state, deterministic: true, shielded: true)
        @action_space.scale(choose_action(state, deterministic: deterministic, shielded: shielded))
      end

      # Trains for `episodes` episodes of at most `steps_per_episode` steps,
      # updating the networks once per step after the warmup.
      def train(episodes:, steps_per_episode:, **_kwargs)
        episodes.times do
          state = @environment.reset
          totals = Hash.new(0.0)
          episode_reward = 0.0
          episode_length = 0

          steps_per_episode.times do
            base_action = (@steps_taken < @config.warmup_steps) ? Array.new(@action_size) { (rand * 2.0) - 1.0 } : sample_action(state)
            executed_action = shielded(state, base_action, totals)
            next_state, reward, done, info = @environment.step(@action_space.scale(executed_action))
            info ||= {}
            info.fetch(:metrics, {}).each { |key, value| totals[:"episode_#{key}"] += value }
            terminal = done && !info[:truncated]
            @replay.push(flatten(state), base_action, reward.to_f, flatten(next_state), terminal ? 1.0 : 0.0)

            episode_reward += reward
            episode_length += 1
            state = next_state
            @steps_taken += 1
            @metrics.record_all(train_step) if @replay.size >= @config.batch_size && @steps_taken >= @config.warmup_steps

            break if done
          end

          @metrics.record(:episode_reward, episode_reward)
          @metrics.record(:episode_length, episode_length)
          totals.each { |key, value| @metrics.record(key, value) }
        end
      end

      # One gradient update of the critics, the actor and alpha on a batch
      # sampled from the replay buffer.
      #
      # @returns [Hash{Symbol => Float}] the losses and current alpha.
      def train_step
        states, actions, rewards, next_states, dones = @replay.sample(@config.batch_size)
        q1_loss, q2_loss = update_critics(states, actions, rewards, next_states, dones)
        policy_loss, safety_loss, entropy = update_policy(states)
        alpha_loss = update_alpha(entropy)
        Reinforce::Networks.soft_update!(target: @q1_target_model, online: @q1_model, tau: @config.tau)
        Reinforce::Networks.soft_update!(target: @q2_target_model, online: @q2_model, tau: @config.tau)

        result = {q1_loss:, q2_loss:, policy_loss:, alpha_loss:, entropy:, alpha: alpha, loss: q1_loss + q2_loss + policy_loss + alpha_loss}
        result[:safety_loss] = safety_loss if @shield
        result
      end

      def replay_size
        @replay.size
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

      # @returns [Float] the current entropy coefficient.
      def alpha
        @log_alpha ? @log_alpha.exp.item : @config.entropy_coefficient
      end

      # A fixed-capacity replay buffer of tensors, overwriting the oldest
      # transitions once full.
      # @private
      class ReplayBuffer
        attr_reader :size

        def initialize(capacity, state_size, action_size)
          @capacity = capacity
          @states = Torch.zeros(capacity, state_size)
          @actions = Torch.zeros(capacity, action_size)
          @rewards = Torch.zeros(capacity)
          @next_states = Torch.zeros(capacity, state_size)
          @dones = Torch.zeros(capacity)
          @position = 0
          @size = 0
        end

        def push(state, action, reward, next_state, done)
          @states[@position] = Torch.tensor(state, dtype: :float32)
          @actions[@position] = Torch.tensor(action, dtype: :float32)
          @rewards[@position] = reward
          @next_states[@position] = Torch.tensor(next_state, dtype: :float32)
          @dones[@position] = done
          @position = (@position + 1) % @capacity
          @size = [@size + 1, @capacity].min
        end

        def sample(batch_size)
          indices = Torch.randint(@size, [batch_size], dtype: :int64)
          [@states, @actions, @rewards, @next_states, @dones].map { |tensor| tensor.index_select(0, indices) }
        end
      end

      private

      def build_network(input_size, output_size)
        Reinforce::Networks.mlp(input_size, output_size, hidden_size: @config.hidden_size, hidden_layers: @config.hidden_layers)
      end

      # The target critics start as copies of the online ones.
      def target_network(online, target, custom:)
        if custom && !target
          raise ArgumentError, "custom critics need target models of the same shape (q1_target_model:, q2_target_model:)"
        end

        target ||= build_network(@state_size + @action_size, 1)
        target.load_state_dict(online.state_dict)
        target
      end

      def flatten(state)
        state.is_a?(Array) ? state.flatten.map(&:to_f) : [state.to_f]
      end

      def state_tensor(state)
        Torch.tensor([flatten(state)], dtype: :float32)
      end

      def choose_action(state, deterministic:, shielded: true)
        action = deterministic ? deterministic_action(state) : sample_action(state)
        shielded ? shielded(state, action, nil) : action
      end

      def deterministic_action(state)
        Torch.no_grad do
          mean, _log_std = gaussian(state_tensor(state))
          Torch.tanh(mean)[0].to_a
        end
      end

      def sample_action(state)
        Torch.no_grad { squashed_sample(state_tensor(state))[0][0].to_a }
      end

      def shielded(state, action, totals)
        return action unless @shield

        executed = @shield.call(state, action)
        if totals && @shield.respond_to?(:last_info)
          @shield.last_info.each { |key, value| totals[:"shield_#{key}"] += value }
        end
        executed
      end

      def gaussian(states)
        mean, log_std = @policy_model.call(states).chunk(2, 1)
        [mean, log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)]
      end

      # Samples tanh-squashed actions with the reparameterization trick.
      #
      # @returns [Array(Torch::Tensor, Torch::Tensor)] the actions (batch x
      #   action components) and their log densities (batch).
      def squashed_sample(states)
        mean, log_std = gaussian(states)
        noise = Torch.randn_like(mean)
        pre_tanh = mean + (log_std.exp * noise)
        gaussian_log_prob = ((noise**2) * -0.5) - log_std - (0.5 * Math.log(2.0 * Math::PI))
        # log(1 - tanh(x)^2), computed stably as 2 * (log 2 - x - softplus(-2x)).
        tanh_correction = (Torch::NN::Functional.softplus(pre_tanh * -2.0) + pre_tanh - Math.log(2.0)) * -2.0
        [Torch.tanh(pre_tanh), (gaussian_log_prob - tanh_correction).sum(1)]
      end

      def q_values(model, states, actions)
        model.call(Torch.cat([states, actions], 1)).reshape(-1)
      end

      def update_critics(states, actions, rewards, next_states, dones)
        target = Torch.no_grad do
          next_actions, next_log_probs = squashed_sample(next_states)
          next_q = Torch.min(q_values(@q1_target_model, next_states, next_actions), q_values(@q2_target_model, next_states, next_actions))
          rewards + (Torch.rsub(dones, 1.0) * (next_q - (next_log_probs * alpha)) * @config.discount_factor)
        end

        q1_loss = Torch::NN::Functional.mse_loss(q_values(@q1_model, states, actions), target)
        q2_loss = Torch::NN::Functional.mse_loss(q_values(@q2_model, states, actions), target)
        @q_optimizer.zero_grad
        (q1_loss + q2_loss).backward
        @q_optimizer.step
        [q1_loss.item, q2_loss.item]
      end

      def update_policy(states)
        actions, log_probs = squashed_sample(states)
        q = Torch.min(q_values(@q1_model, states, actions), q_values(@q2_model, states, actions))
        loss = ((log_probs * alpha) - q).mean
        safety_loss = nil
        if @shield
          mean, _log_std = gaussian(states)
          safety_loss = @shield.safety_loss(states, Torch.tanh(mean))
          loss += safety_loss * @config.safety_weight
        end

        @policy_optimizer.zero_grad
        loss.backward
        @policy_optimizer.step
        [loss.item, safety_loss&.item, -log_probs.mean.item]
      end

      def update_alpha(entropy)
        return 0.0 unless @log_alpha

        alpha_loss = (@log_alpha * (entropy - @target_entropy)).mean
        @alpha_optimizer.zero_grad
        alpha_loss.backward
        @alpha_optimizer.step
        alpha_loss.item
      end
    end
  end
end
