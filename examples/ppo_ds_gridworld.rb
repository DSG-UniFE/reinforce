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
episodes = 1_000
max_actions_per_episode = 250

# Create the dummy representation for the environment

environment = Reinforce::Algorithms::DummyVectorizedEnvironment.new(environment, 1)

# Create the agent
agent = Reinforce::Algorithms::PPODS.new(environment, learning_rate)



warn "Policy Model: #{agent.agent.policy_model}"

# Train the agent
agent.train(episodes, max_actions_per_episode)


agent.save('ds_gridworld_ppo.pth')
File.write('ds_gridworld_ppo_logs.conf', agent.logs.to_s)
# put in eval mode
agent.eval
# Print the learned policy
=begin
puts 'Learned Policy'
plot = UnicodePlot.lineplot(Reinforce.moving_average(agent.logs[:loss], 25), title: "Loss", width: 100, height: 20)
plot.render
plot = UnicodePlot.lineplot(Reinforce.moving_average(agent.logs[:episode_reward], 25), title: "Rewards", width: 100, height: 20)
plot.render
plot = UnicodePlot.lineplot(Reinforce.moving_average(agent.logs[:episode_length], 25), title: "Episode Length", width: 100, height: 20)
plot.render
=end

begin
10.times do
  state = environment.reset
  moves = 0
  max_actions_per_episode.times do
    moves += 1
    action = agent.predict(state)
    #warn "action: #{action} #{action.to_i}"
    state, _, done = environment.step(action.to_i)
    #warn "State: #{state}, Action: #{environment.actions[action]}"
    #environment.render($stdout)
    if done
      warn 'Goal reached! In moves: ' + moves.to_s + 'moves'
      break
    end 
  end
end
end

