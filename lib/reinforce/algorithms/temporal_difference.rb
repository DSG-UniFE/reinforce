# frozen_string_literal: true

module Reinforce
  module Algorithms
    # Generic Temporal Difference learner for discrete action spaces.
    # By default it performs Q-learning; set on_policy: true to perform
    # SARSA-style updates using next_action.
    class TemporalDifference
      attr_reader :q_table, :learning_rate, :discount_factor, :epsilon

      def initialize(environment, learning_rate: 0.1, discount_factor: 0.9, epsilon: 0.1)
        @environment = environment
        @learning_rate = learning_rate
        @discount_factor = discount_factor
        @epsilon = epsilon
        @q_table = Hash.new { |hash, key| hash[key] = Hash.new(0.0) }
      end

      def choose_action(state)
        return actions.sample if rand < @epsilon

        action_values = q_table[state_key(state)]
        actions.max_by { |action| action_values[action] || 0.0 }
      end

      def predict(state)
        action_values = q_table[state_key(state)]
        actions.max_by { |action| action_values[action] || 0.0 }
      end

      def learn(state, action, reward, next_state, done: false, next_action: nil, on_policy: false)
        target = td_target(next_state, reward, done: done, next_action: next_action, on_policy: on_policy)
        key = state_key(state)
        td_error = target - q_table[key][action]
        q_table[key][action] += @learning_rate * td_error
      end

      def td_target(next_state, reward, done:, next_action:, on_policy:)
        return reward.to_f if done

        next_key = state_key(next_state)
        bootstrap = if on_policy && !next_action.nil?
          q_table[next_key][next_action]
        else
          actions.map { |action| q_table[next_key][action] }.max || 0.0
        end
        reward.to_f + @discount_factor * bootstrap
      end

      def reset
        @q_table.clear
      end

      private

      def actions
        @environment.actions
      end

      def state_key(state)
        state.is_a?(Array) ? state.dup.freeze : state
      end
    end
  end
end
