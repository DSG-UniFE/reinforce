#!/usr/bin/env ruby
# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi.

require_relative '../lib/reinforce/q_function_ann'
require_relative '../lib/reinforce/environments/taxiv2'
require_relative '../lib/reinforce/algorithms/dqn'
require 'torch'
require 'forwardable'

# Create the environment
environment = Reinforce::Environments::TaxiV2.new
state_size = environment.state_size
num_actions = environment.actions.size

# Parameters
# Train the agent
learning_rate = 0.01
discount_factor = 0.9
max_actions_per_episode = 250
epsilon = 0.8

warn "State size: #{state_size} actions: #{num_actions}"

# Create the agent
agent = Reinforce::Algorithms::DQN.new(environment)

# Save the model
agent.load('taxi_dqn.pth')

# Print the learned policy

testing_episodes = 100
accomplished = 0
testing_episodes.times do |i|
state = environment.reset
  max_actions_per_episode.times do |_|
    action = agent.predict(state)
    state, reward, done = environment.step(action)
    #puts "Action: #{environment.actions[action.to_i]}, Reward: #{reward}"
    environment.render
    if done
      puts "Task Completed!"
      accomplished += 1
      break
    end
  end
end
print "Accomplished: #{accomplished}/#{testing_episodes} episodes."
