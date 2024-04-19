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
objects = 2
environment = Reinforce::Environments::DSGridWorld.new(size, start, goal, objects)

state = environment.reset
warn "Example State: #{state}"
warn "State Size: #{environment.state_size} #{state[0].length}"

# Parameters
learning_rate = 2.5e-4
episodes = 1000
batch_size = 128

# Create the dummy representation for the environment

environment = Reinforce::DummyVectorizedEnvironment.new(environment, 1)

# Create the agent
agent = Reinforce::Algorithms::PPODS.new(environment, learning_rate)

# Train the agent
agent.train(episodes, batch_size)
agent.save('ds_gridworld_ppo.pth')

puts 'Learned Policy'
plot = UnicodePlot.lineplot(Reinforce.moving_average(agent.logs[:loss], 25), title: "Loss", width: 100, height: 20)
plot.render
plot = UnicodePlot.lineplot(Reinforce.moving_average(agent.logs[:episode_reward], 25), title: "Rewards", width: 100, height: 20)
plot.render
plot = UnicodePlot.lineplot(Reinforce.moving_average(agent.logs[:episode_length], 25), title: "Episode Length", width: 100, height: 20)
plot.render

File.write('ds_gridworld_ppo_logs.logs', agent.logs.to_s)
# put in eval mode
agent.eval

10.times do
  state = environment.reset
  moves = 0
  batch_size.times do
    moves += 1
    action = agent.predict(state)
    #warn "action: #{action} #{action.to_i}"
    state, _, done = environment.step(action.to_i)
    #warn "State: #{state}, Action: #{environment.actions[action.to_i]}"
    if done[0] == true
      warn 'Goal reached! In moves: ' + moves.to_s + 'moves'
      environment.render($stdout)
      break
    end 
  end
end


