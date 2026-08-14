# frozen_string_literal: true

module Reinforce
  module Offline
    module Evaluation
      def self.weighted_importance_sampling(episodes, discount_factor: 1.0, epsilon: 1e-8)
        weighted_returns = []
        weights = []

        episodes.each do |episode|
          discount = 1.0
          episode_return = 0.0
          rho = 1.0

          episode.each do |transition|
            episode_return += discount * transition[:reward].to_f
            behavior_prob = transition.fetch(:behavior_prob, nil)
            target_prob = transition.fetch(:target_prob, nil)
            raise ArgumentError, 'behavior_prob is required for WIS' if behavior_prob.nil?
            raise ArgumentError, 'target_prob is required for WIS' if target_prob.nil?

            rho *= target_prob.to_f / [behavior_prob.to_f, epsilon].max
            discount *= discount_factor.to_f
          end

          weighted_returns << rho * episode_return
          weights << rho
        end

        denominator = weights.sum
        return 0.0 if denominator.abs < epsilon

        weighted_returns.sum / denominator
      end

      # Fitted Q Evaluation for deterministic target policies on discrete actions.
      class TabularFQE
        attr_reader :q_values

        def initialize(dataset, discount_factor: 0.99, iterations: 50)
          @dataset = dataset
          @discount_factor = discount_factor
          @iterations = iterations
          @q_values = Hash.new(0.0)
        end

        def fit(policy)
          @iterations.times do
            targets = Hash.new { |hash, key| hash[key] = [] }

            @dataset.each do |transition|
              state_key = state_key(transition[:state])
              action = transition[:action]
              reward = transition[:reward].to_f
              done = transition[:done]

              target = if done
                reward
              else
                next_action = policy.call(transition[:next_state])
                reward + @discount_factor * @q_values[[state_key(transition[:next_state]), next_action]]
              end

              targets[[state_key, action]] << target
            end

            targets.each do |key, values|
              @q_values[key] = values.sum / values.size.to_f
            end
          end
          self
        end

        def policy_value(policy, initial_states)
          return 0.0 if initial_states.empty?

          values = initial_states.map do |state|
            action = policy.call(state)
            @q_values[[state_key(state), action]]
          end
          values.sum / values.size.to_f
        end

        private

        def state_key(state)
          state.is_a?(Array) ? state.dup.freeze : state
        end
      end
    end
  end
end
