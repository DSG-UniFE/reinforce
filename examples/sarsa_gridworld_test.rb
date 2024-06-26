#!/usr/bin/env ruby
# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi and Filippo Poltronieri.

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
max_actions_per_episode = 150

# Create the agent
# The q_function is loaded from file
learning_rate = 0.01
discount_factor = 0.7
q_function_model = Reinforce::QFunctionANN.new(state_size, num_actions, learning_rate, discount_factor)

agent = Reinforce::Algorithms::SARSA.new(environment, q_function_model, 0.5)
agent.load('gridworld_sarsa.pth')

# Print the learned policy
puts 'Learned Policy'
testing_episodes = 100
accomplished = 0
testing_episodes.times do |i|
  puts '----------------'
  state = environment.reset
  max_actions_per_episode.times do
    action = agent.predict(state)
    state, _, done = environment.step(action.to_i)
    #environment.render($stdout)
    if done
      puts "Goal reached! Episode #{i}"
      accomplished += 1
      break
    end
  end
end

puts "Accomplished: #{accomplished}/#{testing_episodes} episodes."
