#!/usr/bin/env ruby
# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi and Filippo Poltronieri.
require 'reinforce'
require 'torch'
require 'forwardable'
require 'unicode_plot'

# Create the environment, Initialize the number of obstacles
size = 10
start = [0, 0]
goal = [size - 1, size - 1]

obstacles = 4

environment = Reinforce::Environments::DSGridWorld.new(size, start, goal, obstacles)

state = environment.reset
warn "Example State: #{state}"
warn "State Size: #{environment.state_size} #{environment.state_size[1]}"

# Parameters
learning_rate = 1e-5
episodes = 225
batch_size = 512

# Create the dummy representation for the environment

environment = Reinforce::DummyVectorizedEnvironment.new(environment, 1)

# Create the agent
agent = Reinforce::Algorithms::PPODS.new(environment, learning_rate)

# Train the agent
agent.train(episodes, batch_size)
agent.save('ds_gridworld_ppo.pth')

begin
  puts 'Learned Policy'
  plot = UnicodePlot.lineplot(Reinforce.moving_average(agent.logs[:loss], 25), title: "Loss", width: 100, height: 20)
  plot.render
  plot = UnicodePlot.lineplot(Reinforce.moving_average(agent.logs[:episode_reward], 25), title: "Rewards", width: 100, height: 20)
  plot.render
  plot = UnicodePlot.lineplot(Reinforce.moving_average(agent.logs[:episode_length], 25), title: "Episode Length", width: 100, height: 20)
  plot.render
rescue StandardError => e
  warn "Error: #{e}"
end

File.write('ds_gridworld_ppo_logs.logs', agent.logs.to_s)
# put in eval mode
agent.eval

10.times do
  state = environment.reset
  mask = environment.action_masks
  moves = 0
  batch_size.times do
    moves += 1
    action = agent.predict(state, mask)
    state, _, done = environment.step(action.to_i)
    mask = environment.action_masks
    if done[0] == true
      #warn 'Goal reached! In moves: ' + moves.to_s + 'moves'
      #environment.render($stdout)
      break
    end
  end
end


