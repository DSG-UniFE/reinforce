# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi.

require_relative '../experience'
require_relative '../categorical_distribution'

module Reinforce
  module Algorithms
    ##
    # This class implements a Reinforcement Learning agent that uses the
    # n-step SARSA algorithm with an epsilon greedy policy.
    class SARSA
      def initialize(environment, q_function_model, epsilon = 0.01)
        @environment = environment
        @q_function_model = q_function_model
        @initial_epsilon = epsilon
        # Create a store for learning experience
        @experience = ::Reinforce::Experience.new
      end

      def choose_action(state, epsilon)
        # Choose action according to the policy, with epsilon greedy algorithm
        # for governing the exploration / exploitation trade-off.
        if epsilon > rand
          @q_function_model.random_action(state)
        else
          # Obtain the logits of each action from the model
          logits = @q_function_model.forward(state)

          # Return greedy action from the distribution
          CategoricalDistribution.new(logits: logits).greedy
        end
      end

      def predict(state)
        # Return the action to be taken according to the policy learnt
        @q_function_model.get_action(state)
      end

      # Train the agent.
      #
      # @param num_episodes [Integer] the number of episodes to consider
      # @param batch_size [Integer] the number of actions that the agent takes
      # in each episode (note that the agent might reach the goal state before
      # this number is reached: in that case, the episode terminates)
      # @return [void]
      def train(num_episodes, batch_size)
        # Epsilon greedy algorithm implements a dynamic exploration /
        # exploitation tradeoff. The epsilon parameter starts at the initial
        # value and decays over the training process to reach zero at the end
        # of it.
        epsilon = @initial_epsilon

        # Training loop
        1.upto(num_episodes) do |episode_number|
          #puts "Episode: #{episode_number} epsilon: #{epsilon}"
          progress = episode_number.to_f / num_episodes * 100
          print "\rTraining: #{progress.round(2)}%"
          # Reset the environment
          state = @environment.reset

          action = choose_action(state, epsilon)
          # Setup number of actions to take before updating the Q function
          actions_left = batch_size - 1

          # Episode loop
          loop do
            # Choose an action

            # Take the action and observe the next state and reward
            next_state, reward, done = @environment.step(action.to_i)
            actions_left -= 1

            next_action = choose_action(state, epsilon)
            # Update the agent
            @experience.update(state.dup, action, next_state.dup, next_action, reward, done)
            
            state = next_state
            action = next_action

            break if done || actions_left.zero? # Reached the goal state
          end

          # Update Q function after each batch of actions
          @q_function_model.update(@experience)

          # Decay epsilon
          epsilon = @initial_epsilon * (num_episodes - episode_number) / num_episodes

          # Reset learning experience
          @experience.reset
        end
      end

      # Save the model
      def save(path)
        @q_function_model.save(path)
      end

      # Load the model from a file
      def load(path)
        @q_function_model.load(path)
      end

    end
  end
end
