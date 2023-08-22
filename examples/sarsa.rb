#!/usr/bin/env ruby
# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi.

require_relative '../lib/reinforce/algorithms/sarsa'
require_relative '../lib/reinforce/environments/gridworld'
require 'torch'
require 'forwardable'

# Create the environment
size = 20
start = [0, 0]
goal = [size - 1, size - 1]
obstacles = Array.new(10) { |_| [1 + rand(size - 2), 1 + rand(size - 2)] }
environment = Reinforce::Environments::GridWorld.new(size, start, goal, obstacles)
state_size = environment.state_size
num_actions = environment.actions.size

# warn "state_size: #{state_size.inspect}"

# Parameters
learning_rate = 0.01
discount_factor = 0.7
episodes = 5000
max_actions_per_episode = 100

# input to the network is the current state
# output of the network is the log probabilities of each action
class QFunctionANN
  extend Forwardable

  def_delegators :@architecture, :forward, :parameters

  def initialize(state_size, num_actions)
    @num_actions = num_actions
    @architecture = Torch::NN::Sequential.new(
      Torch::NN::Linear.new(state_size, 512),
      Torch::NN::ReLU.new,
      Torch::NN::Linear.new(512, 512),
      Torch::NN::ReLU.new,
      Torch::NN::Linear.new(512, num_actions)
    )
    @architecture.train # Enable training mode
  end

  def random_action(_state)
    rand(@num_actions)
  end
end

# Create the policy model
q_function_model = QFunctionANN.new(state_size, num_actions)

# Create the optimizer
optimizer = Torch::Optim::Adam.new(q_function_model.parameters, lr: learning_rate)

# Create the agent
agent = Reinforce::Algorithms::Sarsa.new(discount_factor, q_function_model, optimizer)

# Training loop
1.upto(episodes) do |episode_number|
  puts "Episode: #{episode_number}"
  # Reset the environment
  init_state = environment.reset
  state = Torch::Tensor.new(init_state)
  actions_left = max_actions_per_episode

  # Choose an action
  action_tplus1 = agent.choose_action(state)

  # warn "action_tplus1: #{action_tplus1.inspect}"

  # Episode loop
  loop do
    action_t = action_tplus1

    # Take the action and observe the next state and reward
    state_tplus1, reward, done = environment.step(action_t)
    # warn "state_tplus1: #{state_tplus1.inspect}"

    # Update the agent
    agent.update(state, action_t, state_tplus1, reward, done)

    actions_left -= 1

    # Choose an action
    action_tplus1 = agent.choose_action(Torch::Tensor.new(state_tplus1))
    # warn "action_tplus1: #{action_tplus1.inspect}"

    break if done || actions_left.zero? # Reached the goal state
  end

  # Update Q function after each batch of actions
  agent.update_q_function
  agent.reset
end

# Print the learned policy
puts 'Learned Policy'
(0...size).each do |i|
  (0...size).each do |j|
    action = agent.choose_action(Torch::Tensor.new([i, j]))
    puts "State [#{i},#{j}]: Action #{action}"
  end
end
