#!/usr/bin/env ruby
# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi, Filippo Poltronieri.

require 'reinforce'
require 'torch'
require 'forwardable'

# Create the environment
environment = Reinforce::Environments::Game2048.new()
state_size = environment.state_size
# up, down, right, left (is there really a difference?)
num_actions = environment.actions.size

puts "State size #{state_size}, action_size: #{num_actions}"

# Parameters
learning_rate = 0.01
discount_factor = 0.7
epsilon = 0.8
episodes = 15_000
max_actions_per_episode = 500

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
    max_actions_per_episode.times do |i|
      # Choose an action
      action = agent.predict(state)
      #puts "Action: #{action}" 
      # Take the action and observe the next state and reward
      next_state, reward, done = environment.step(action)
      # Update the agent
      state = next_state

      #environment.render($stdout)
      #sleep(0.5)
      if done
        puts "Episode ended after #{i} steps"
        break
      end 
    end
    environment.render($stdout)
    avg_score << environment.score
  end

  puts "Average score #{avg_score.sum / avg_score.length}"
end
