# frozen_string_literal: true

require "torch"
require_relative "../hyperparameters"
require_relative "../networks"
require_relative "../environment"
require_relative "ppo"

module Reinforce
  module Algorithms
    # DIAYN, "Diversity Is All You Need" (Eysenbach et al., 2018): learns a
    # set of distinct skills with no reward at all. Each episode runs one
    # skill z, drawn uniformly from `skills`; the policy sees the state plus
    # a one-hot encoding of z; and a discriminator q(z | s) learns to tell
    # from a state which skill reached it. The policy's only reward is
    #
    #   log q(z | s') - log p(z)
    #
    # high when the state it reached identifies its skill. Skills are
    # therefore pushed apart, each towards states the others don't visit --
    # behaviors that can later be selected between, or fine-tuned, for a
    # real task.
    #
    # The DIAYN-specific parts live in a SkillEnvironment wrapper (skill
    # sampling, observation augmentation, reward replacement, discriminator
    # training), so the policy is learned by an ordinary algorithm on that
    # wrapper, PPO by default. An on-policy learner keeps the rewards in
    # step with the discriminator; a replay-based one (SAC) would learn from
    # rewards computed by older discriminators.
    class DIAYN
      include ::Reinforce::Agent

      # Hyperparameters (see Reinforce::Hyperparameters).
      #
      # skills: how many skills to learn.
      # discriminator_learning_rate / discriminator_network: the
      #   discriminator's optimizer and shape.
      # discriminator_updates / discriminator_batch_size: gradient steps (on
      #   random minibatches of recent (state, skill) pairs) at the end of
      #   each episode.
      # discriminator_buffer_size: how many recent pairs to keep.
      # ignore_done: keep running a skill until the step limit even when the
      #   environment signals done, so a skill isn't cut short by stumbling
      #   on a task's goal (DIAYN has no task).
      Config = Data.define(
        :skills, :discriminator_learning_rate, :discriminator_network, :discriminator_updates,
        :discriminator_batch_size, :discriminator_buffer_size, :ignore_done
      ) do
        def initialize(skills: 8, discriminator_learning_rate: 1e-3, discriminator_network: Reinforce::Networks::Config.new,
          discriminator_updates: 4, discriminator_batch_size: 64, discriminator_buffer_size: 10_000, ignore_done: true)
          super
        end
      end

      attr_reader :config, :metrics, :agent, :discriminator, :skill_environment

      # @parameter agent [#call] builds the policy learner for the
      #   skill-conditioned environment it is given; PPO by default.
      def initialize(environment, config = nil, agent: nil, **hyperparameters)
        @config = Reinforce::Hyperparameters.resolve(Config, config, hyperparameters)
        @environment = environment
        @metrics = Reinforce::Metrics::Recorder.new
        @discriminator = @config.discriminator_network.build(environment.state_size, @config.skills)
        @discriminator_optimizer = Torch::Optim::Adam.new(@discriminator.parameters, lr: @config.discriminator_learning_rate)
        @skill_environment = SkillEnvironment.new(environment, self)
        @agent = (agent || ->(env) { PPO.new(env) }).call(@skill_environment)
      end

      # Trains skills and discriminator together, reward-free. Episodes
      # and steps mean whatever they mean for the underlying agent.
      def train(episodes:, steps_per_episode:, **kwargs)
        @agent.train(episodes:, steps_per_episode:, **kwargs)
        self
      end

      # The action skill `skill` takes in `state` (an observation of the
      # wrapped environment, without the skill).
      def predict(state, deterministic: true, skill: 0)
        @agent.predict(with_skill(state, skill), deterministic:)
      end

      # @returns [Array(Float)] q(z | state) for every skill z.
      def skill_probabilities(state)
        Torch.no_grad { Torch.softmax(@discriminator.call(state_tensor([state])), dim: 1) }.to_a.first
      end

      # The observation the agent sees: `state` followed by a one-hot skill.
      def with_skill(state, skill)
        raise ArgumentError, "skill must be in 0...#{@config.skills}" unless skill.is_a?(Integer) && skill.between?(0, @config.skills - 1)

        one_hot = Array.new(@config.skills, 0.0)
        one_hot[skill] = 1.0
        (state.respond_to?(:to_a) ? state.to_a : state).map(&:to_f) + one_hot
      end

      # DIAYN's reward for reaching `state` while running `skill`:
      # log q(skill | state) - log p(skill), with p uniform.
      def skill_reward(state, skill)
        log_probabilities = Torch.no_grad { Torch.log_softmax(@discriminator.call(state_tensor([state])), dim: 1) }
        log_probabilities[0][skill].item + Math.log(@config.skills)
      end

      # One round of discriminator training on (state, skill) pairs.
      def train_discriminator(states, skills)
        logits = @discriminator.call(state_tensor(states))
        targets = Torch.tensor(skills, dtype: :int64)
        loss = Torch::NN::Functional.cross_entropy(logits, targets)
        @discriminator_optimizer.zero_grad
        loss.backward
        @discriminator_optimizer.step
        accuracy = logits.argmax(1).eq(targets).to(:float32).mean.item
        @metrics.record_all({discriminator_loss: loss.item, discriminator_accuracy: accuracy})
      end

      private

      def state_tensor(states)
        Torch.tensor(states.map { |state| (state.respond_to?(:to_a) ? state.to_a : state).map(&:to_f) }, dtype: :float32)
      end

      # Wraps the task environment for DIAYN: a uniformly random skill per
      # episode, the skill appended to every observation, DIAYN's reward in
      # place of the environment's (kept in info[:extrinsic_reward]), and
      # discriminator training at the end of each episode.
      class SkillEnvironment
        include ::Reinforce::Environment

        attr_reader :skill

        def initialize(environment, diayn)
          @environment = environment
          @diayn = diayn
          @config = diayn.config
          @states = []
          @skills = []
        end

        def reset
          train_discriminator if @episode_steps&.positive?
          @skill = rand(@config.skills)
          @episode_steps = 0
          @diayn.with_skill(@environment.reset, @skill)
        end

        def step(action)
          next_state, reward, done, info = @environment.step(action)
          remember(next_state)
          @episode_steps += 1
          diayn_reward = @diayn.skill_reward(next_state, @skill)
          @diayn.metrics.record(:skill_reward, diayn_reward)
          [@diayn.with_skill(next_state, @skill), diayn_reward, done && !@config.ignore_done,
            (info || {}).merge(extrinsic_reward: reward, skill: @skill)]
        end

        def state_size = @environment.state_size + @config.skills

        def actions = @environment.actions

        def render(*arguments) = @environment.render(*arguments)

        private

        def remember(state)
          @states << (state.respond_to?(:to_a) ? state.to_a : state)
          @skills << @skill
          return unless @states.size > @config.discriminator_buffer_size

          @states.shift
          @skills.shift
        end

        def train_discriminator
          @config.discriminator_updates.times do
            picks = Array.new([@config.discriminator_batch_size, @states.size].min) { rand(@states.size) }
            @diayn.train_discriminator(picks.map { |index| @states[index] }, picks.map { |index| @skills[index] })
          end
        end
      end
    end
  end
end
