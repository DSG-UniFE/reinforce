#!/usr/bin/env ruby
# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2024, by Mauro Tortonesi and Filippo Poltronieri.

require 'reinforce'
require 'forwardable'

environment = Reinforce::Environments::OneDimensionalWalk.new(10)

agent = Reinforce::Algorithms::DQN.new(environment)

max_actions_per_episode = 50 

agent.load('1dwalk_dqn.pth')

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
      warn 'Goal reached!' + "in #{moves} moves.\n"
      break
    end
  end
end


