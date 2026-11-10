# frozen_string_literal: true

require_relative "../environment"
require_relative "../metrics"

module Reinforce
  module Wrappers
    # Measures exploration: records every distinct state the wrapped
    # environment returns from #reset and #step, whichever agent is driving
    # it. At the end of each episode (when #step returns done, or at the
    # next #reset) it records into its own #metrics:
    #
    # - :states_visited -- distinct states seen so far, over all episodes;
    # - :state_coverage -- that count as a fraction of the reachable states,
    #   when they are known (see `reachable_states:` below);
    #
    # with the total number of environment steps taken so far as the step,
    # so curves from different agents line up on the same x-axis.
    #
    # Other methods (e.g. GridWorld#obstacles) pass through to the wrapped
    # environment.
    class StateCoverage
      include ::Reinforce::Environment

      attr_reader :environment, :metrics, :steps_taken, :reachable_states

      # @parameter reachable_states [Array, nil] the states that count
      #   towards full coverage; defaults to the environment's own
      #   #reachable_states when it has one (as GridWorld does).
      def initialize(environment, reachable_states: nil)
        @environment = environment
        @reachable_states = reachable_states || (environment.reachable_states if environment.respond_to?(:reachable_states))
        @visited = Set.new
        @steps_taken = 0
        @episode_open = false
        @metrics = Reinforce::Metrics::Recorder.new
      end

      def reset
        close_episode
        state = @environment.reset
        visit(state)
        @episode_open = true
        state
      end

      def step(action)
        next_state, reward, done, info = @environment.step(action)
        @steps_taken += 1
        visit(next_state)
        close_episode if done
        [next_state, reward, done, info]
      end

      def state_size = @environment.state_size

      def actions = @environment.actions

      def render(*arguments) = @environment.render(*arguments)

      # @returns [Integer] distinct states seen so far.
      def states_visited
        @visited.size
      end

      # @returns [Float, nil] states_visited as a fraction of the
      #   reachable states, or nil if those aren't known.
      def coverage
        return nil if @reachable_states.nil?

        states_visited.fdiv(@reachable_states.size)
      end

      def respond_to_missing?(name, include_private = false)
        @environment.respond_to?(name, include_private) || super
      end

      def method_missing(name, ...)
        @environment.respond_to?(name) ? @environment.public_send(name, ...) : super
      end

      private

      def visit(state)
        @visited << (state.respond_to?(:to_a) ? state.to_a : state)
      end

      def close_episode
        return unless @episode_open

        @episode_open = false
        @metrics.record(:states_visited, states_visited, step: @steps_taken)
        @metrics.record(:state_coverage, coverage, step: @steps_taken) if @reachable_states
      end
    end
  end
end
