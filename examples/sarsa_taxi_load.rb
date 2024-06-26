#!/usr/bin/env ruby
# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi and Filippo Poltronieri.

require 'reinforce'
require 'torch'
require 'forwardable'

# Create the environment
environment = Reinforce::Environments::TaxiV2.new
state_size = environment.state_size
num_actions = environment.actions.size

# Parameters
# Train the agent
learning_rate = 0.01
discount_factor = 0.7
episodes = 10_000
max_actions_per_episode = 100
epsilon = 0.8

warn "State size: #{state_size} actions: #{num_actions}"

# Create the Q function: we are using a neural network model for it
q_function_model = Reinforce::QFunctionANN.new(state_size, num_actions, learning_rate, discount_factor)

# Create the agent
agent = Reinforce::Algorithms::SARSA.new(environment, q_function_model, epsilon)

agent.load('taxi_sarsa.pth')

# Print the learned policy
state = environment.reset
puts 'Learned Policy'
testing_episodes = 100
accomplished = 0
testing_episodes.times do |i|
  max_actions_per_episode.times do |_|
      action = agent.predict(state)
      state, reward, done = environment.step(action.to_i)
      environment.render
      if done
        puts "Task Completed!"
        accomplished += 1
        break
      end
  end
end
puts "Accomplished: #{accomplished}/#{testing_episodes} episodes."
