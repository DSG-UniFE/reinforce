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
objects = 2
environment = Reinforce::Environments::Pacman.new(size, start, objects)

state = environment.reset
warn "Example State: #{state}"
warn "State Size: #{environment.state_size} #{state[0].length}"

# Parameters
learning_rate = 0.001
max_actions_per_episode = 250

# Create the dummy representation for the environment

environment = Reinforce::DummyVectorizedEnvironment.new(environment, 1)

# Create the agent
agent = Reinforce::Algorithms::PPODS.new(environment, learning_rate)


# Train the agent
agent.load('ds_pacman_ppo.pth')

# put in eval mode
agent.eval

win = 0
10_000.times do
  state = environment.reset
  mask = environment.action_masks
  moves = 0
  max_actions_per_episode.times do
    moves += 1
    action = agent.predict(state, mask)
    #warn "action: #{action} #{action.to_i}"
    state, _, done = environment.step(action.to_i)
    mask = environment.action_masks
    if done[0] == true
      #warn 'Goal reached! In moves: ' + moves.to_s + 'moves'
      #environment.render($stdout)
      win += 1
      break
    end 
  end
end

warn "Win: #{win}/10 episodes."
