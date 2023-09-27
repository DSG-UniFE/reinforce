# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi.

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
        @log_probs = []
        @rewards = []
      end

      def reset
        @log_probs = []
        @rewards = []
      end

      def choose_action(state)
        # Obtain the log probabilities of each action from the model
        logits = @model.forward(state)

        # Sample an action from the distribution
        pd = CategoricalDistribution.new(logits: logits.to_a)
        action = pd.sample

        # Store the log probability of the action
        @log_probs << pd.log_probability(action)

        action
      end

      def get_reward(reward)
        @rewards << reward
      end

      def update_policy
        discounted_rewards = calculate_discounted_rewards(@rewards)
        rewards = Torch::Tensor.new(discounted_rewards)
        log_probs = Torch::Tensor.new(@log_probs)
        loss = Torch.sum(- log_probs * rewards)
        @optimizer.zero_grad
        @optimizer.step(proc { loss })
        loss
      end

      # save the model at a given path
      def save(path)
        @model.save(path)
      end
      
      # load the model at the given path
      def load(path)
        @model.load(path)
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
