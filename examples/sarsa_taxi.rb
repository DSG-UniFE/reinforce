#!/usr/bin/env ruby
# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi.

require 'reinforce'
require 'torch'
require 'forwardable'

# Create the environment
environment = Reinforce::Environments::Taxi.new
state_size = environment.state_size
num_actions = environment.actions.size

# Parameters
# Train the agent
learning_rate = 0.01
discount_factor = 0.7
episodes = 5_000
max_actions_per_episode = 150
epsilon = 0.8

warn "State size: #{state_size} actions: #{num_actions}"

# Create the Q function: we are using a neural network model for it
q_function_model = Reinforce::QFunctionANN.new(state_size, num_actions, learning_rate, discount_factor)

# Create the agent
agent = Reinforce::Algorithms::SARSA.new(environment, q_function_model, epsilon)

# Train the agent
agent.train(episodes, max_actions_per_episode)

# Save the model
agent.save('taxi_sarsa.pth')

# Print the learned policy
testing_episodes = 10
testing_episodes.times do
  state = environment.reset
  max_actions_per_episode.times do |i|
      action = agent.predict(state)
      state, reward, done = environment.step(action.to_i)
      #environment.render
      if done
        puts "Task Completed! in #{i} steps"
        break
      end
  end
end
