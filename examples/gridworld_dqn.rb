#!/usr/bin/env ruby
# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi.

require_relative '../lib/reinforce/q_function_ann'
require_relative '../lib/reinforce/algorithms/dqn'
require_relative '../lib/reinforce/environments/gridworld'
require 'torch'
require 'forwardable'

# Create the environment
size = 20
start = [0, 0]
goal = [size - 1, size - 1]
obstacles = Array.new(10) { |_| [1 + rand(size - 2), 1 + rand(size - 2)] }
environment = Reinforce::Environments::GridWorld.new(size, start, goal, obstacles)
state_size = environment.state_size
num_actions = environment.actions.size

# Parameters
learning_rate = 0.01
discount_factor = 0.7
episodes = 1000
max_actions_per_episode = 100

# Create the Q function: we are using a neural network model for it
q_function_model = Reinforce::QFunctionANN.new(state_size, num_actions, learning_rate, discount_factor)
q_function_model_target = Reinforce::QFunctionANN.new(state_size, num_actions, learning_rate, discount_factor)

# Create the agent
agent = Reinforce::Algorithms::DQN.new(environment, q_function_model, q_function_model_target)

# Train the agent
agent.train(episodes, max_actions_per_episode)

# Print the learned policy
puts 'Learned Policy'
(0...size).each do |i|
  (0...size).each do |j|
    action = agent.choose_action(Torch::Tensor.new([i, j]))
    puts "State [#{i},#{j}]: Action #{action}"
  end
end
