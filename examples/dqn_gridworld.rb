#!/usr/bin/env ruby
# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi.
require 'reinforce'
require_relative '../lib/reinforce/q_function_ann'
require_relative '../lib/reinforce/algorithms/dqn'
require_relative '../lib/reinforce/environments/gridworld'
require 'torch'
require 'forwardable'
require 'unicode_plot'

# Create the environment
size = 10
start = [0, 0]
goal = [size - 1, size - 1]
obstacles = 5
environment = Reinforce::Environments::GridWorld.new(size, start, goal, obstacles)
state_size = environment.state_size
num_actions = environment.actions.size

# Parameters
learning_rate = 0.001
discount_factor = 0.99
episodes = 1_500
max_actions_per_episode = 150


# Create the agent
agent = Reinforce::Algorithms::DQN.new(environment, learning_rate, discount_factor)

# Train the agent
agent.train(episodes, max_actions_per_episode)
agent.save('gridworld_dqn.pth')
# Print the learned policy

plot = UnicodePlot.lineplot(Reinforce.moving_average(agent.logs[:loss], 25), title: "Loss", width: 100, height: 20)
plot.render
plot = UnicodePlot.lineplot(Reinforce.moving_average(agent.logs[:episode_reward], 25), title: "Rewards", width: 100, height: 20)
plot.render
plot = UnicodePlot.lineplot(Reinforce.moving_average(agent.logs[:episode_length], 25), title: "Episode Length", width: 100, height: 20)
plot.render

puts 'Learned Policy'
state = environment.reset 
max_actions_per_episode.times do |i|
  action = agent.predict(state) 
  state, _, done = environment.step(action.to_i)
  #environment.render($stdout)
  if done
    warn 'Goal reached in ' + i.to_s + ' steps.'
    break
  end 
end
