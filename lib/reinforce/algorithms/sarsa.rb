# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi.

require 'torch'
require_relative '../categorical_distribution'

module Reinforce
  module Algorithms
    class Sarsa
      def initialize(discount_factor, q_function_model, optimizer, epsilon = 0.01)
        @discount_factor = discount_factor
        @epsilon = epsilon
        @q_function_model = q_function_model
        @optimizer = optimizer
        reset
      end

      def reset
        @history = { state: [], action: [], next_state: [], reward: [], done: [] }
      end

      def history_size
        @history[:state].size
      end

      def choose_action(state)
        # Choose action according to the policy, with epsilon greedy algorithm
        # for governing the exploration / exploitation trade-off.
        if @epsilon > rand
          @q_function_model.random_action(state)
        else
          # Obtain the log probabilities of each action from the model
          logits = @q_function_model.forward(state)

          # Return greedy action from the distribution
          CategoricalDistribution.new(logits: logits.to_a).greedy
        end
      end

      def update(state, action, next_state, reward, done)
        @history[:state] << state
        @history[:action] << action
        @history[:next_state] << next_state
        @history[:reward] << reward
        @history[:done] << done
      end

      def update_q_function
        next_actions = @history[:next_state].map do |next_state|
          # Need to tell Torch not to track the gradient for these operations.
          # See L. Graesser, W.L. Keng, "Foundations of Deep Reinforcement
          # Learning", Section 3.5.2, page 70.
          Torch.no_grad { @q_function_model.forward(Torch::Tensor.new(next_state)).argmax.to_i }
        end
        target_actions = next_actions
          .zip(@history[:reward], @history[:done]).map do |next_action, reward, done|
          if done
            reward
          else
            reward + @discount_factor * next_action
          end
        end
        criterion = Torch::NN::MSELoss.new
        @optimizer.zero_grad
        warn "target_actions: #{target_actions.inspect}"
        warn "@history[:action]: #{@history[:action].inspect}"
        loss = criterion.call(Torch::Tensor.new(target_actions), Torch::Tensor.new(@history[:action]))
        @optimizer.step(proc { loss })
      end
    end
  end
end
