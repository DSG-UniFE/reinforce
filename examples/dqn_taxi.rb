#!/usr/bin/env ruby
# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi and Filippo Poltronieri.

require 'torch'
require 'forwardable'
require 'unicode_plot'
require 'reinforce'

# Create the environment
environment = Reinforce::Environments::Taxi.new
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

plot = UnicodePlot.lineplot(Reinforce.moving_average(agent.logs[:loss], 25), title: "Loss", width: 100, height: 20)
plot.render
plot = UnicodePlot.lineplot(Reinforce.moving_average(agent.logs[:episode_reward], 25), title: "Rewards", width: 100, height: 20)
plot.render
plot = UnicodePlot.lineplot(Reinforce.moving_average(agent.logs[:episode_length], 25), title: "Episode Length", width: 100, height: 20)
plot.render


# Print the learned policy
state = environment.reset
puts 'Learned Policy'
max_actions_per_episode.times do |_|
  action = agent.predict(state)
  state, reward, done = environment.step(action.to_i)
  #environment.render
  if done
    puts "Task Completed!"
    break
  end
end
