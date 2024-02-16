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
size = 10
start = [0, 0]
goal = [size - 1, size - 1]
obstacles = 5
environment = Reinforce::Environments::GridWorld.new(size, start, goal, obstacles)
state_size = environment.state_size
num_actions = environment.actions.size

# Parameters
learning_rate = 0.01
discount_factor = 0.7
max_actions_per_episode = 150

# Create the Q function: we are using a neural network model for i
q_function_model = Reinforce::QFunctionANN.new(state_size, num_actions, learning_rate, discount_factor)
q_function_model_target = Reinforce::QFunctionANN.new(state_size, num_actions, learning_rate, discount_factor)

# Create the agent
agent = Reinforce::Algorithms::DQN.new(environment, q_function_model, q_function_model_target)

# Load the agent

agent.load('gridworld_dqn.pth')

# Print the learned policy
puts 'Exploiting the learned Policy'
test_episodes = 100
accomplished = 0
test_episodes.times do 
  state = environment.reset 
  max_actions_per_episode.times do
    action = agent.predict(state) 
    #warn "State: #{state} action: #{action}"
    state, _, done = environment.step(action.to_i)
    #environment.render($stdout)
    if done
      warn 'Goal reached!'
      accomplished += 1
      break
    end 
  end
end

puts "Accomplished: #{accomplished}/#{test_episodes} episodes."
