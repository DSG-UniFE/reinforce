#!/usr/bin/env ruby
# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2024, by Mauro Tortonesi and Filippo Poltronieri.

require 'reinforce'
require 'forwardable'
require 'unicode_plot'

environment = Reinforce::Environments::OneDimensionalWalk.new(10)

agent = Reinforce::Algorithms::DQN.new(environment, 0.01, 0.7)

max_actions_per_episode = 50 

agent.train(1000, max_actions_per_episode)

agent.save('1dwalk_dqn.pth')

plot = UnicodePlot.lineplot(Reinforce.moving_average(agent.logs[:loss], 25), title: 'Loss', width: 100, height: 20)
plot.render
plot = UnicodePlot.lineplot(Reinforce.moving_average(agent.logs[:episode_reward], 25), title: 'Rewards', width: 100, height: 20)
plot.render
plot = UnicodePlot.lineplot(Reinforce.moving_average(agent.logs[:episode_length], 25), title: 'Episode Length', width: 100, height: 20)
plot.render

testing_episodes = 10

testing_episodes.times do |i|
  moves = 0
  state = environment.reset
  max_actions_per_episode.times do
    action = agent.predict(state)
    state, _, done = environment.step(action.to_i)
    environment.render($stdout)
    moves += 1
    if done
      warn 'Goal reached!' + "in #{moves} moves."
      break
    end
  end
end


