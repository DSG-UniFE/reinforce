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
goal = [size - 1, size - 1]
obstacles = 5
environment = Reinforce::Environments::GridWorld.new(size, start, goal, obstacles)
state_size = environment.state_size
num_actions = environment.actions.size

# warn "state_size: #{state_size.inspect}"

# Parameters
learning_rate = 0.01
discount_factor = 0.99
episodes = 5000
max_actions_per_episode = 150

# input to the network is the current state
# output of the network is the log probabilities of each action
class NNPolicy
  extend Forwardable

  def_delegators :@architecture, :forward, :parameters

  def initialize(state_size, num_actions)
    @architecture = Torch::NN::Sequential.new(
      Torch::NN::Linear.new(state_size, 512),
      Torch::NN::ReLU.new,
      Torch::NN::Linear.new(512, 512),
      Torch::NN::ReLU.new,
      Torch::NN::Linear.new(512, num_actions)
    )
    @architecture.train # Enable training mode
  end
end

# Create the policy model
model = NNPolicy.new(state_size, num_actions)

# Create the optimizer
optimizer = Torch::Optim::Adam.new(model.parameters, lr: learning_rate)

# Create the agent
agent = Reinforce::Algorithms::Reinforce.new(state_size, num_actions, discount_factor, model, optimizer)

# Training loop
1.upto(episodes) do |episode_number|
  puts "Episode: #{episode_number}"
  # Reset the environment
  init_state = environment.reset
  state = Torch::Tensor.new(init_state)
  actions_left = max_actions_per_episode

  # Episode loop
  loop do
    # Choose an action
    action = agent.choose_action(state)

    # Take the action and observe the next state and reward
    next_state, reward, done = environment.step(action)

    actions_left -= 1

    agent.get_reward(reward)

    # Update the current state
    state = Torch::Tensor.new(next_state)

    break if done || actions_left.zero? # Reached the goal state
  end

  # Update policy at the end of each episode
  agent.update_policy
end

# Print the learned policy
puts 'Learned Policy'
2.times do |i|
  warn "Starting episode #{i}"
  state = environment.reset
  max_actions_per_episode.times do
    action = agent.predict(Torch::Tensor.new(state))
    state, _, done = environment.step(action)
    #warn "State: #{state} Action: #{action} Done: #{done}"
    if done
      puts 'Goal reached!'
      break
    end
  end
end
