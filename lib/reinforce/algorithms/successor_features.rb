# frozen_string_literal: true

require "torch"
require_relative "../hyperparameters"
require_relative "../networks"
require_relative "../categorical_distribution"

module Reinforce
  module Algorithms
    # Successor features with generalized policy improvement (Barreto et
    # al., 2017): learn once, from features alone, how a few policies behave,
    # then act well on *new* tasks with no further training.
    #
    # The environment reports a feature vector phi for every transition, in
    # `info[:features]`, and a task is a weight vector w with reward
    # phi . w (see MultiGoalGridWorld). For a policy pi, its successor
    # features are
    #
    #   psi(s, a) = E[ sum_t gamma^t phi_t | s, a, then follow pi ]
    #
    # so its Q-values on *any* task w are psi(s, a) . w. Training learns one
    # psi network per training task w_i, for the policy that is greedy with
    # respect to psi_i . w_i -- the SFQL scheme from the paper: each
    # episode's behavior follows one training task (epsilon-greedy, in
    # turn), and every transition updates *all* psi_i off-policy. The
    # environment's own reward is never used.
    #
    # On a new task w, #predict acts by generalized policy improvement:
    #
    #   argmax_a max_i psi_i(s, a) . w
    #
    # which is guaranteed to be at least as good as every training policy
    # evaluated on w, and is often much better when w combines them.
    class SuccessorFeatures
      include ::Reinforce::Agent

      # Hyperparameters (see Reinforce::Hyperparameters).
      #
      # epsilon / final_epsilon: exploration rate, decayed linearly from the
      #   first to the second over each #train call.
      # warmup_steps: transitions collected before any update.
      # tau: Polyak averaging factor for the target networks.
      # network: the psi networks' shape.
      Config = Data.define(
        :learning_rate, :discount_factor, :epsilon, :final_epsilon, :batch_size, :replay_size,
        :warmup_steps, :tau, :network
      ) do
        def initialize(learning_rate: 1e-3, discount_factor: 0.95, epsilon: 1.0, final_epsilon: 0.05, batch_size: 64,
          replay_size: 50_000, warmup_steps: 500, tau: 0.01, network: Reinforce::Networks::Config.new)
          super
        end
      end

      attr_reader :config, :metrics, :tasks, :psi_networks

      # @parameter tasks [Array(Array(Numeric))] the training tasks: weight
      #   vectors over the environment's features. Their policies are what
      #   generalized policy improvement combines on new tasks.
      def initialize(environment, config = nil, tasks:, **hyperparameters)
        raise ArgumentError, "at least one training task is required" if tasks.empty?

        @config = Reinforce::Hyperparameters.resolve(Config, config, hyperparameters)
        @environment = environment
        @actions = environment.actions
        @tasks = tasks.map { |task| task.map(&:to_f) }
        @feature_size = @tasks.first.size
        output_size = @actions.size * @feature_size
        @psi_networks = @tasks.map { @config.network.build(environment.state_size, output_size) }
        @target_networks = @psi_networks.map do |network|
          @config.network.build(environment.state_size, output_size).tap { |target| target.load_state_dict(network.state_dict) }
        end
        @optimizers = @psi_networks.map { |network| Torch::Optim::Adam.new(network.parameters, lr: @config.learning_rate) }
        @replay = []
        @steps_taken = 0
        @metrics = Reinforce::Metrics::Recorder.new
      end

      def train(episodes:, steps_per_episode:, **_kwargs)
        episodes.times do |episode|
          epsilon = @config.epsilon + (@config.final_epsilon - @config.epsilon) * episode.fdiv([episodes - 1, 1].max)
          task_index = episode % @tasks.size
          state = @environment.reset
          task_return = 0.0
          length = 0

          steps_per_episode.times do
            action_index = (rand < epsilon) ? rand(@actions.size) : greedy_action(state, [task_index], @tasks[task_index])
            next_state, _reward, done, info = @environment.step(action_index)
            features = info&.fetch(:features) { raise ArgumentError, "the environment must report info[:features]" }
            store(state, action_index, features, next_state, done)
            task_return += dot(features, @tasks[task_index])
            length += 1
            @steps_taken += 1
            update if @steps_taken >= @config.warmup_steps && @replay.size >= @config.batch_size
            state = next_state
            break if done
          end

          @metrics.record_all({episode_reward: task_return, episode_length: length, epsilon:})
        end
        self
      end

      # The action generalized policy improvement chooses on `task` (the
      # first training task by default): the best action of the best
      # training policy, as judged by task. With deterministic: false, an
      # action sampled from a softmax over those GPI values.
      def predict(state, deterministic: true, task: nil)
        values = q_values(state, task: task || @tasks.first)
        return @actions[values.index(values.max)] if deterministic

        @actions[CategoricalDistribution.new(logits: Torch.tensor(values)).choose(deterministic: false)]
      end

      # GPI action values on `task`: max over training policies i of
      # psi_i(state, a) . task, for each action a.
      def q_values(state, task:, policies: (0...@tasks.size))
        weights = Torch.tensor(task.map(&:to_f), dtype: :float32)
        Torch.no_grad do
          policies.map { |index| psi(@psi_networks[index], state_tensor([state]))[0].matmul(weights) }
            .then { |values| Torch.stack(values).max(0).first } # torch-rb's max(dim) returns [values, indices]
        end.to_a
      end

      private

      def greedy_action(state, policies, task)
        values = q_values(state, task:, policies:)
        values.index(values.max)
      end

      # [batch, actions, features]
      def psi(network, states)
        network.call(states).reshape(states.size(0), @actions.size, @feature_size)
      end

      def store(state, action_index, features, next_state, done)
        @replay << [plain(state), action_index, features.map(&:to_f), plain(next_state), done ? 1.0 : 0.0]
        @replay.shift while @replay.size > @config.replay_size
      end

      # One TD update of every psi_i on a shared minibatch, each towards
      # phi + gamma * psi_i_target(s', a'_i), where a'_i is the action
      # policy i (greedy on task i) takes in s'.
      def update
        batch = Array.new(@config.batch_size) { @replay[rand(@replay.size)] }
        states = state_tensor(batch.map { |transition| transition[0] })
        actions = Torch.tensor(batch.map { |transition| transition[1] }, dtype: :int64)
        features = Torch.tensor(batch.map { |transition| transition[2] }, dtype: :float32)
        next_states = state_tensor(batch.map { |transition| transition[3] })
        not_done = 1.0 - Torch.tensor(batch.map { |transition| transition[4] }, dtype: :float32)
        index = actions.reshape(-1, 1, 1).expand(-1, 1, @feature_size)

        losses = @psi_networks.each_with_index.map do |network, task_index|
          weights = Torch.tensor(@tasks[task_index], dtype: :float32)
          target = Torch.no_grad do
            next_psi = psi(@target_networks[task_index], next_states)
            next_actions = psi(network, next_states).matmul(weights).argmax(1)
            next_index = next_actions.reshape(-1, 1, 1).expand(-1, 1, @feature_size)
            features + @config.discount_factor * not_done.reshape(-1, 1) * next_psi.gather(1, next_index).squeeze(1)
          end
          prediction = psi(network, states).gather(1, index).squeeze(1)
          loss = (prediction - target).pow(2).mean
          @optimizers[task_index].zero_grad
          loss.backward
          @optimizers[task_index].step
          ::Reinforce::Networks.soft_update!(target: @target_networks[task_index], online: network, tau: @config.tau)
          loss.item
        end
        @metrics.record(:loss, losses.sum / losses.size)
      end

      def dot(features, task)
        features.zip(task).sum { |feature, weight| feature * weight }
      end

      def plain(state)
        (state.respond_to?(:to_a) ? state.to_a : state).map(&:to_f)
      end

      def state_tensor(states)
        Torch.tensor(states.map { |state| plain(state) }, dtype: :float32)
      end
    end
  end
end
