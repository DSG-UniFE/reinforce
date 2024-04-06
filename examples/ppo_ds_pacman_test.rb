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
objects = 5
environment = Reinforce::Environments::Pacman.new(size, start, objects)

state = environment.reset
warn "Example State: #{state}"
warn "State Size: #{environment.state_size} #{state[0].length}"

# Parameters
learning_rate = 0.001
episodes = 1_000
max_actions_per_episode = 250

# Create the dummy representation for the environment

environment = Reinforce::DummyVectorizedEnvironment.new(environment, 1)

# Create the agent
agent = Reinforce::Algorithms::PPODS.new(environment, learning_rate)


# Train the agent
agent.load('ds_pacman_ppo.pth')

File.write('ds_pacman_ppo_logs.conf', agent.logs.to_s)

# put in eval mode
agent.eval

10.times do
  state = environment.reset
  moves = 0
  max_actions_per_episode.times do
    moves += 1
    action = agent.predict(state)
    #warn "action: #{action} #{action.to_i}"
    state, _, done = environment.step(action.to_i)
    if done[0] == true
      warn 'Goal reached! In moves: ' + moves.to_s + 'moves'
      environment.render($stdout)
      break
    end 
    if moves == max_actions_per_episode
      warn 'Max moves reached'
      environment.render($stdout)
      break
    end
  end
end


