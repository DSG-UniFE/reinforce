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
episodes = 10_000
max_actions_per_episode = 250
epsilon = 0.8

warn "State size: #{state_size} actions: #{num_actions}"

# Create the agent
agent = Reinforce::Algorithms::DQN.new(environment, learning_rate, discount_factor) 

# Train the agent
agent.train(episodes, max_actions_per_episode)
# Save the model
agent.save('taxi_dqn.pth')

# Print the learned policy
state = environment.reset
puts 'Learned Policy'
max_actions_per_episode.times do |_|
  action = agent.predict(state)
  state, reward, done = environment.step(action)
  puts "Action: #{environment.actions[action]}, Reward: #{reward}"
  environment.render
  if done
    puts "Task Completed!"
    break
  end
end
