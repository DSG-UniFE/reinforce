#!/usr/bin/env ruby
# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi, Filippo Poltronieri.

require_relative '../lib/reinforce/q_function_ann'
require_relative '../lib/reinforce/algorithms/sarsa'
require_relative '../lib/reinforce/environments/game_2048'
require 'torch'
require 'forwardable'

# Create the environment
environment = Reinforce::Environments::Game2048.new()

state_size = environment.state_size
size = 4 # board size is 4, meaning that I would have a 4x4 board

# up, down, right, left (is there really a difference?)
num_actions = environment.actions.size

puts "State size #{state_size}, action_size: #{num_actions}"

# Parameters
learning_rate = 0.01
discount_factor = 0.7
epsilon = 0.2
episodes = 5_000
max_actions_per_episode = 100

# Create the Q function: we are using a neural network model for it
q_function_model = Reinforce::QFunctionANN.new(state_size, num_actions, learning_rate, discount_factor)

# Create the agent
agent = Reinforce::Algorithms::SARSA.new(environment, q_function_model, epsilon)

# Train the agent
agent.train(episodes, max_actions_per_episode)

puts "Training done! Start the exploitation"

agent.save('2048_network_rev2.pth')

state_size = environment.state_size
# up, down, right, left (is there really a difference?)
num_actions = environment.actions.size

puts "State size #{state_size}, action_size: #{num_actions}"


begin
  # Print the learned policy
  #
  puts 'Learned Policy -- starting to exploit'
  # Reset the environment

  # Episode loop -- Exploit the training
  avg_score = []
  1.upto(5) do |i|
    puts "Episode #{i}"
    state = environment.reset
    j = 1
    loop do
      # Choose an action
      action = agent.choose_action(state)
      #puts "Action: #{action}" 
      # Take the action and observe the next state and reward
      next_state, reward, done = environment.step(action)
      # Update the agent
      state = next_state

      j += 1
      #environment.render($stdout)
      #sleep(0.5)
      puts "done!" if done
      break if done  # Reached the goal state
    end
    environment.render($stdout)
    avg_score << environment.score
  end

  puts "Average score #{avg_score.sum / avg_score.length}"
end
