#!/usr/bin/env ruby
# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi.

require_relative '../lib/reinforce/q_function_ann'
require_relative '../lib/reinforce/algorithms/ppo'
require_relative '../lib/reinforce/environments/gridworld'
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
episodes = 2500
max_actions_per_episode = 150

# Create the agent
agent = Reinforce::Algorithms::PPO.new(environment, learning_rate)

# Train the agent
agent.train(episodes, max_actions_per_episode)


agent.save('gridworld_ppo.pth')

# put in eval mode
agent.eval
# Print the learned policy
puts 'Learned Policy'

10.times do
  state = environment.reset
  max_actions_per_episode.times do
    action = agent.predict(state)
    #warn "action: #{action} #{action.to_i}"
    state, _, done = environment.step(action.to_i)
    #warn "State: #{state}, Action: #{environment.actions[action]}"
    environment.render($stdout)
    if done
      warn 'Goal reached!'
      break
    end 
  end
end