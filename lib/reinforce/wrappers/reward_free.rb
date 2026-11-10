# frozen_string_literal: true

require_relative "../environment"

module Reinforce
  module Wrappers
    # Hides an environment's reward: #step always returns a reward of 0.0,
    # with the original reward moved to `info[:extrinsic_reward]` so
    # evaluation code can still read it. Use it to make reward-free
    # pretraining explicit -- an agent trained on the wrapped environment
    # demonstrably never saw the task's reward. Termination is unchanged.
    #
    # Other methods (e.g. GridWorld#reachable_states) pass through to the
    # wrapped environment.
    class RewardFree
      include ::Reinforce::Environment

      attr_reader :environment

      def initialize(environment)
        @environment = environment
      end

      def reset = @environment.reset

      def state_size = @environment.state_size

      def actions = @environment.actions

      def render(*arguments) = @environment.render(*arguments)

      def step(action)
        next_state, reward, done, info = @environment.step(action)
        [next_state, 0.0, done, (info || {}).merge(extrinsic_reward: reward)]
      end

      def respond_to_missing?(name, include_private = false)
        @environment.respond_to?(name, include_private) || super
      end

      def method_missing(name, ...)
        @environment.respond_to?(name) ? @environment.public_send(name, ...) : super
      end
    end
  end
end
