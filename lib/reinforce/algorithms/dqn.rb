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
      def initialize(environment, q_function_model, q_function_model_target, epsilon = 0.9)
        @environment = environment
        @q_function_model = q_function_model
        @q_function_model_target = q_function_model_target
        # Create prioritized experience replay store
        @prioritized_experience_replay = PrioritizedExperienceReplay.new
        # tau is the Polyak averaging parameter, it should be between 0 and 1
        @tau = 1.0
        @initial_epsilon = epsilon
        @training_start = 1000
        @update_frequency_for_q = 10
        @update_frequency_for_q_target = 500
        @optimizer = Torch::Optim::Adam.new(@q_function_model.parameters, lr: 0.001)
        @discount_factor = 0.99
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
          #CategoricalDistribution.new(logits: logits).greedy
          logits.argmax.to_i
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

        total_steps = num_episodes * batch_size
        
        # Epsilon greedy algorithm implements a dynamic exploration /
        # exploitation tradeoff. The epsilon parameter starts at the initial
        # value and decays over the training process to reach zero at the end
        # of it.
        epsilon = @initial_epsilon

        minibatch_size = 128
        global_step = 0

        state = @environment.reset
        actions_left = batch_size

        # Training loop
        1.upto(total_steps) do 
          # warn "Episode: #{episode_number}"
          progress = global_step.to_f / total_steps * 100
          print "\rTraining: #{progress.round(2)}%"
          # Reset the environment

          # Setup number of actions to take before updating the Q function

          # Choose an action, according to epsilon-greedy policy
          action = choose_action(state, epsilon)

          # Take the action and observe the next state and reward
          next_state, reward, done = @environment.step(action.to_i)
          actions_left -= 1

          # Store the experience in the replay memory
          @prioritized_experience_replay.update(state, action, next_state, reward, done)

          # Update the count of steps taken so far
          global_step += 1

          if global_step > @training_start
            # Sample a minibatch of experiences from the replay memory
            # Update Q function every @update_frequency_for_q steps
            if (global_step % @update_frequency_for_q).zero?
              experience = @prioritized_experience_replay.sample(minibatch_size)
              target = nil
              Torch.no_grad do
                target_max = @q_function_model_target.architecture.call(Torch.tensor(experience[:next_state], dtype: :float32)).max#.max(dim: 1)
                t_rewards = Torch.tensor(experience[:reward])
                dones = experience[:done].map { |d| d ? 0 : 1 }
                target = t_rewards + @discount_factor * target_max * (1- Torch.tensor(dones))
              end
              t_actions = Torch.tensor(experience[:action])
              old_val = @q_function_model.forward(experience[:state])
              told_val = Torch.zeros_like(t_actions, dtype: :float32)
              old_val.zip(t_actions).each_with_index do |(val, action), i|
                told_val[i] = val[action]
              end
              criterion = Torch::NN::MSELoss.new
              loss = criterion.call(told_val, target)
              #warn "Loss: #{loss}"
              @optimizer.zero_grad
              loss.backward
              @optimizer.step
              #@q_function_model.update(experience)
            end

            # Soft-update target Q function every @update_frequency_for_q_target steps
            if (global_step % @update_frequency_for_q_target).zero?
              @q_function_model_target.soft_update(@q_function_model, @tau)
            end
          end

          state = next_state

          if done || actions_left.zero? # Reached the goal state
            actions_left = batch_size
            state = @environment.reset
          end

          # Decay epsilon
          epsilon = @initial_epsilon * (total_steps - global_step) / total_steps
        end

      end     

      def predict(state)
        # Return the action to be taken according to the policy
        @q_function_model.get_action(state)
      end
      
    # Save the model after training_start

      def save(path)
        @q_function_model.save(path)
      end
        
        # load the model if a file already exists
      def load(path)
        @q_function_model.load(path)
      end

    end
  end
end
