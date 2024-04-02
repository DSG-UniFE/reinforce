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
obstacles = 5
environment = Reinforce::Environments::GridWorld.new(size, start, goal, obstacles)

# Parameters
learning_rate = 0.001
episodes = 3_000
max_actions_per_episode = 150

# Create the agent
# Define a policy model for the agent 
policy = Torch::NN::Sequential.new(
  Torch::NN::Linear.new(2, 64),
  Torch::NN::Tanh.new,
  Torch::NN::Linear.new(64, 64),
  Torch::NN::Tanh.new,
  Torch::NN::Linear.new(64, environment.actions.size))
value = Torch::NN::Sequential.new(
  Torch::NN::Linear.new(2, 64),
  Torch::NN::Tanh.new,
  Torch::NN::Linear.new(64, 1))

agent = Reinforce::Algorithms::PPO.new(environment, learning_rate, policy, value)

# Train the agent
agent.train(episodes, max_actions_per_episode)


agent.save('gridworld_ppo.pth')

# put in eval mode
agent.eval
# Print the learned policy
puts 'Learned Policy'
plot = UnicodePlot.lineplot(Reinforce.moving_average(agent.logs[:loss], 25), title: "Loss", width: 100, height: 20)
plot.render
plot = UnicodePlot.lineplot(Reinforce.moving_average(agent.logs[:episode_reward], 25), title: "Rewards", width: 100, height: 20)
plot.render
plot = UnicodePlot.lineplot(Reinforce.moving_average(agent.logs[:episode_length], 25), title: "Episode Length", width: 100, height: 20)
plot.render


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

