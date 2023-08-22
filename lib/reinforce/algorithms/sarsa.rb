# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi.

require_relative '../experience'
require_relative '../categorical_distribution'

module Reinforce
  module Algorithms
    class Sarsa
      def initialize(environment, q_function_model, epsilon = 0.01)
        @environment = environment
        @q_function_model = q_function_model
        @epsilon = epsilon
        # Create a store for learning experience
        @experience = ::Reinforce::Experience.new
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

      def train(episodes, batch_size)
        # Training loop
        1.upto(episodes) do |episode_number|
          puts "Episode: #{episode_number}"
          # Reset the environment
          state = @environment.reset

          # Setup number of actions to take before updating the Q function
          actions_left = batch_size

          # Episode loop
          loop do
            # Choose an action
            action = choose_action(state)

            # Take the action and observe the next state and reward
            next_state, reward, done = @environment.step(action)
            actions_left -= 1

            # Update the agent
            @experience.update(state, action, next_state, reward, done)

            state = next_state

            break if done || actions_left.zero? # Reached the goal state
          end

          # Update Q function after each batch of actions
          @q_function_model.update(@experience)

          # Reset learning experience
          @experience.reset
        end
      end
    end
  end
end
