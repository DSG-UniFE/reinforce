#!/usr/bin/env ruby
# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi and Filippo Poltronieri.

require 'torch'
require 'reinforce'
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

architectureq = Torch::NN::Sequential.new(
          Torch::NN::Linear.new(state_size, 128),
          Torch::NN::ReLU.new,
          Torch::NN::Linear.new(128, 128),
          Torch::NN::ReLU.new,
          Torch::NN::Linear.new(128, num_actions)
        )

q_function_model = Reinforce::QFunctionANN.new(environment.state_size, environment.actions.size, 
                                              learning_rate, discount_factor, architecture: architectureq)

# We do not need to load the target network because when testing

agent = Reinforce::Algorithms::DQN.new(environment, learning_rate, discount_factor, 
                                       q_function_model: q_function_model, q_function_model_target: nil)

# Load the agent parameters
agent.load('gridworld_dqn.pth')

# Print the learned policy
puts 'Exploiting the learned Policy'
test_episodes = 100
accomplished = 0
test_episodes.times do 
  state = environment.reset 
  max_actions_per_episode.times do |s|
    action = agent.predict(state) 
    #warn "State: #{state} action: #{action}"
    state, _, done = environment.step(action.to_i)
    #environment.render($stdout)
    if done && s < max_actions_per_episode - 1
      warn 'Goal reached!'
      accomplished += 1
      break
    end 
  end
end

puts "Accomplished: #{accomplished}/#{test_episodes} episodes."
