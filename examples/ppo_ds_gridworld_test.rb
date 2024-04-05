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
objects = 5
environment = Reinforce::Environments::DSGridWorld.new(size, start, goal, objects)

state = environment.reset
warn "Example State: #{state}"
warn "State Size: #{environment.state_size} #{state[0].length}"

# Parameters
learning_rate = 0.001

max_actions_per_episode = 500

# Create the dummy representation for the environment

environment = Reinforce::Algorithms::DummyVectorizedEnvironment.new(environment, 1)

# Create the agent
agent = Reinforce::Algorithms::PPODS.new(environment, learning_rate)

warn "Policy Model: #{agent.agent.policy_model}"

# Train the agent

agent.load('ds_gridworld_ppo.pth')
agent.eval

10.times do
  state = environment.reset
  moves = 0
  max_actions_per_episode.times do
    moves += 1
    action = agent.predict(state)
    state, _, done = environment.step(action.to_i)
    #warn "State: #{state}, Action: #{environment.actions[action.to_i]} done: #{done}"
    #environment.render($stdout)
    if done.first == true
      warn 'Goal reached! In moves: ' + moves.to_s + ' moves' + ' State: ' + state.to_s
      environment.render($stdout)
      break
    end 
  end

end
