#!/usr/bin/env ruby
# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi.

require_relative '../lib/reinforce/q_function_ann'
require_relative '../lib/reinforce/algorithms/sarsa'
require_relative '../lib/reinforce/environments/taxi'
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
max_actions_per_episode = 100
epsilon = 0.6

warn "State size: #{state_size} actions: #{num_actions}"

# Create the Q function: we are using a neural network model for it
q_function_model = Reinforce::QFunctionANN.new(state_size, num_actions, learning_rate, discount_factor)

# Create the agent
agent = Reinforce::Algorithms::SARSA.new(environment, q_function_model, epsilon)

# Train the agent
agent.train(episodes, max_actions_per_episode)


# Print the learned policy
state = environment.reset
puts 'Learned Policy'
100.times do |_|
    action = agent.predict(state)
    puts "Action: #{environment.actions[action]}"
    state, _, done = environment.step(action)
    environment.render
    if done
      puts "Task Completed!"
      break
    end
end
