# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi.

require_relative "../experience"
require_relative "../categorical_distribution"
require_relative "../prioritized_experience_replay"
require_relative "../q_function_ann"
require_relative "../models/dueling_q_network"

module Reinforce
  module Algorithms
    ##
    # Deep Q-Network (Mnih et al., 2015): epsilon-greedy action selection,
    # a target network for stable bootstrapping, and prioritized experience
    # replay (Schaul et al., 2015). The actual TD update -- including the
    # importance-sampling-weighted loss and the target/Double-DQN logic
    # below -- is delegated to QFunctionANN#update rather than hand-rolled
    # here; see lib/reinforce/q_function_ann.rb.
    #
    # Two well-established, cheap-to-add improvements over vanilla DQN are
    # available as constructor flags:
    #
    # - `double_dqn: true` (Van Hasselt et al., 2015) selects the bootstrap
    #   action via this network's own argmax but evaluates it using the
    #   target network, removing the overestimation bias vanilla DQN gets
    #   from using the same (noisy) estimator for both.
    # - `dueling: true` (Wang et al., 2016) swaps the Q-network's
    #   architecture for Reinforce::Models::DuelingQNetwork, which learns a
    #   state-value stream and an action-advantage stream separately. Only
    #   takes effect when `q_function_model`/`q_function_model_target`
    #   aren't supplied explicitly -- if you build your own, pass a
    #   DuelingQNetwork-based one directly instead.
    class DQN
      include ::Reinforce::Agent

      attr_reader :logs

      def initialize(environment, learning_rate = 2.5e-4, discount_factor = 0.99, epsilon = 0.9,
        q_function_model: nil, q_function_model_target: nil, double_dqn: false, dueling: false)
        @environment = environment
        @double_dqn = double_dqn
        @q_function_model = q_function_model ||
          build_q_function(environment, learning_rate, discount_factor, dueling)
        @q_function_model_target = q_function_model_target ||
          build_q_function(environment, learning_rate, discount_factor, dueling)
        # Create prioritized experience replay store
        @prioritized_experience_replay = PrioritizedExperienceReplay.new
        # tau is the Polyak averaging parameter, it should be between 0 and 1
        @tau = 1.0
        @learning_rate = learning_rate
        @initial_epsilon = epsilon
        @training_start = 1000
        @update_frequency_for_q = 10
        @update_frequency_for_q_target = 500
        @discount_factor = discount_factor
        @logs = {loss: [], episode_reward: [], episode_length: []}
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
          logits.argmax.to_i
        end
      end

      # Train the agent.
      #
      # @param episodes [Integer] the number of episodes to consider
      # @param steps_per_episode [Integer] the number of actions that the
      # agent takes in each episode (note that the agent might reach the
      # goal state before this number is reached: in that case, the episode
      # terminates)
      # @return [void]
      def train(episodes:, steps_per_episode:, **_kwargs)
        total_steps = episodes * steps_per_episode

        # Epsilon greedy algorithm implements a dynamic exploration /
        # exploitation tradeoff. The epsilon parameter starts at the initial
        # value and decays over the training process to reach zero at the end
        # of it.
        epsilon = @initial_epsilon

        minibatch_size = 128
        global_step = 0

        state = @environment.reset
        actions_left = steps_per_episode
        episode_length = 0
        episode_reward = 0

        # Training loop
        1.upto(total_steps) do
          progress = global_step.to_f / total_steps * 100
          print "\rTraining: #{progress.round(2)}%" if global_step % 100 == 0

          # Choose an action, according to epsilon-greedy policy
          action = choose_action(state, epsilon)

          # Take the action and observe the next state and reward
          next_state, reward, done = @environment.step(action.to_i)
          actions_left -= 1
          episode_length += 1
          episode_reward += reward

          # Store the experience in the replay memory
          @prioritized_experience_replay.update(state, action, next_state, reward, done)

          # Update the count of steps taken so far
          global_step += 1

          if global_step > @training_start
            # Sample a minibatch of experiences from the replay memory
            # Update Q function every @update_frequency_for_q steps
            if (global_step % @update_frequency_for_q).zero?
              experience = @prioritized_experience_replay.sample(minibatch_size)
              result = @q_function_model.update(
                experience,
                target: @q_function_model_target,
                double_dqn: @double_dqn,
                weights: experience[:weights]
              )
              @logs[:loss] << result[:loss]

              # Feed the freshly-observed TD-errors back so future sampling
              # reflects how wrong the Q-function currently is about these
              # transitions.
              @prioritized_experience_replay.update_priorities(experience[:indices], result[:td_errors])
            end

            # Soft-update target Q function every @update_frequency_for_q_target steps
            if (global_step % @update_frequency_for_q_target).zero?
              @q_function_model_target.soft_update(@q_function_model, @tau)
            end
          end

          state = next_state

          if done || actions_left.zero? # Reached the goal state
            actions_left = steps_per_episode
            state = @environment.reset
            @logs[:episode_reward] << episode_reward
            @logs[:episode_length] << episode_length
            episode_reward = 0
            episode_length = 0
          end

          # Decay epsilon
          epsilon = @initial_epsilon * (total_steps - global_step) / total_steps
        end
      end

      def predict(state)
        # Return the action to be taken according to the policy
        @q_function_model.get_action(state)
      end

      # Save the model
      def save(path)
        @q_function_model.save(path)
      end

      # load the model if a file already exists
      def load(path)
        @q_function_model.load(path)
      end

      private

      def build_q_function(environment, learning_rate, discount_factor, dueling)
        architecture = if dueling
          ::Reinforce::Models::DuelingQNetwork.new(environment.state_size, environment.actions.size, hidden_size: 512)
        end
        QFunctionANN.new(environment.state_size, environment.actions.size, learning_rate, discount_factor, architecture: architecture)
      end
    end
  end
end
