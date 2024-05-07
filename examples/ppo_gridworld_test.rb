#!/usr/bin/env ruby
# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi and Filippo Poltronieri.

require 'reinforce'
require 'torch'
require 'forwardable'

# Create the environment
size = 10
start = [0, 0]
goal = [size - 1, size - 1]
obstacles = 5
environment = Reinforce::Environments::GridWorld.new(size, start, goal, obstacles)

# Parameters
learning_rate = 0.001
max_actions_per_episode = 150

# Create the agent
agent = Reinforce::Algorithms::PPO.new(environment, learning_rate)

# Load the agent from file
agent.load('gridworld_ppo.pth')

# put in eval mode
agent.eval
# Print the learned policy

puts 'Exploiting the Learned Policy'

testing_episodes = 100
accomplished = 0
testing_episodes.times do
  state = environment.reset
  max_actions_per_episode.times do |s|
    action = agent.predict(state) 
    #warn "State: #{state} action: #{action}"
    state, _, done = environment.step(action.to_i)
    #environment.render($stdout)
    if done && s < max_actions_per_episode - 1
      accomplished += 1
      break
    end 
  end
end

puts "Accomplished: #{accomplished}/#{testing_episodes} episodes."
