#!/usr/bin/env ruby
# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi and Filippo Poltronieri.
require 'reinforce'
require 'torch'
require 'forwardable'
require 'unicode_plot'

# Create the environment
size = 10
start = [0, 0]
goal = [size - 1, size - 1]
objects = 4
environment = Reinforce::Environments::DSGridWorld.new(size, start, goal, objects)

state = environment.reset
warn "Example State: #{state}"
warn "State Size: #{environment.state_size} #{state[0].length}"

# Parameters
learning_rate = 0.001
max_actions_per_episode = 512

# Create the dummy representation for the environment

environment = Reinforce::DummyVectorizedEnvironment.new(environment, 1)

# Create the agent
agent = Reinforce::Algorithms::PPODS.new(environment, learning_rate)

warn "Policy Model: #{agent.agent.policy_model}"

# Train the agent

agent.load('ds_gridworld_ppo.pth')
agent.eval

win = 0
100.times do
  state = environment.reset
  mask = environment.action_masks
  moves = 0
  max_actions_per_episode.times do
    moves += 1
    action = agent.predict(state, mask, false)
    #warn "Predicted action #{action}"
    state, _, done = environment.step(action.to_i)
    mask = environment.action_masks
    if done.first == true and moves < max_actions_per_episode
      win += 1
      #warn 'Goal reached! In moves: ' + moves.to_s + ' moves' + ' State: ' + state.to_s
      #environment.render($stdout)
      break
    end 
  end

end

warn "Win: #{win}/1000 episodes."

