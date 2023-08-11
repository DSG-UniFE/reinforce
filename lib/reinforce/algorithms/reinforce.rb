# frozen_string_literal: true

require 'torch'
require_relative '../categorical_distribution'

module Reinforce
  module Algorithms
    class Reinforce
      def initialize(num_states, num_actions, discount_factor, model, optimizer)
        @num_states = num_states
        @num_actions = num_actions
        @discount_factor = discount_factor
        @model = model
        @optimizer = optimizer
      end

      def choose_action(state)
        # Obtain the log probabilities of each action from the model
        lp_params = @model.forward(state.flatten)

        # Sample an action from the distribution
        pd = CategoricalDistribution.new(log_probs: lp_params)
        action = pd.sample

        # Store the log probability of the action
        @log_probs << pd.log_probability(action)

        action
      end

      def get_reward(reward)
        @rewards << reward
      end

      def update_policy
        discounted_rewards = Torch::Tensor.new(calculate_discounted_rewards(rewards))
        loss = Torch.sum(-Torch.stack(@log_probs) * discounted_rewards)
        @optimizer.zero_grad
        loss.backward
        @optimizer.step
        loss
      end

      private

      def calculate_discounted_rewards(rewards)
        discounted_rewards = []
        cumulative_reward = 0.0

        rewards.reverse_each do |reward|
          cumulative_reward = reward + @discount_factor * cumulative_reward
          discounted_rewards.unshift(cumulative_reward)
        end

        discounted_rewards
      end
    end
  end
end
