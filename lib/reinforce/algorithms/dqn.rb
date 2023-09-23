# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi.

require_relative '../experience'
require_relative '../categorical_distribution'
require_relative '../prioritized_experience_replay'

module Reinforce
  module Algorithms
    ##
    # This class implements a Reinforcement Learning agent that uses the
    # Deep Q Network algorithm.
    class DQN
      def initialize(environment, q_function_model, q_function_model_target)
        @environment = environment
        @q_function_model = q_function_model
        @q_function_model_target = q_function_model_target
        # Create prioritized experience replay store
        @prioritized_experience_replay = PrioritizedExperienceReplay.new
        # tau is the Polyak averaging parameter, it should be between 0 and 1
        @tau = 1.0
        @initial_epsilon = 0.001
        @training_start = 1000
        @update_frequency_for_q = 100
        @update_frequency_for_q_target = 500
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
          CategoricalDistribution.new(logits: logits.to_a).greedy
        end
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

        minibatch_size = 100
        global_step = 0


        # Training loop
        1.upto(num_episodes) do |episode_number|
          puts "Episode: #{episode_number}"
          # Reset the environment
          state = @environment.reset

          # Setup number of actions to take before updating the Q function
          actions_left = batch_size

          # Perform batch_size steps by acting in the environment, storing the
          # experience in a replay memory, and updating the Q and Q target
          # functions
          batch_size.times do
            # Choose an action, according to epsilon-greedy policy
            action = choose_action(state, epsilon)

            # Take the action and observe the next state and reward
            next_state, reward, done = @environment.step(action)
            actions_left -= 1

            # Store the experience in the replay memory
            @prioritized_experience_replay.update(state, action, next_state, reward, done)

            # Sample a minibatch of experiences from the replay memory
            experience = @prioritized_experience_replay.sample(minibatch_size)

            # Update the count of steps taken so far
            global_step += 1

            if global_step > @training_start
              # Update Q function every @update_frequency_for_q steps
              if (global_step % @update_frequency_for_q).zero?
                @q_function_model.update(experience)
              end

              # Soft-update target Q function every @update_frequency_for_q_target steps
              if (global_step % @update_frequency_for_q_target).zero?
                @q_function_model_target.soft_update(@q_function_model, @tau)
              end
            end

            state = next_state

            break if done || actions_left.zero? # Reached the goal state
          end

          # Decay epsilon
          epsilon = @initial_epsilon * (num_episodes - episode_number) / num_episodes
        end
      end
    end
  end
end
