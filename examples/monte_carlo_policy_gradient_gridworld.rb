#!/usr/bin/env ruby
# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi and Filippo Poltronieri.

require "reinforce"
require "torch"
require "tmpdir"

# Create the environment
size = 10
start = [0, 0]
goal = [size - 1, size - 1]
obstacles = 5
environment = Reinforce::Environments::GridWorld.new(size, start, goal, obstacles)

# Parameters
learning_rate = 1e-4 # 1e-2 collapses the policy onto a fixed path within ~100 episodes
discount_factor = 0.99
episodes = 5000
max_actions_per_episode = 150

# The policy network maps the current state to one logit per action. It is
# a plain Torch::NN::Module, so the agent can #save/#load its state_dict.
model = Reinforce::Networks.mlp(environment.state_size, environment.actions.size, hidden_size: 512)

# Create the optimizer
optimizer = Torch::Optim::Adam.new(model.parameters, lr: learning_rate)

# Create the agent
agent = Reinforce::Algorithms::MonteCarloPolicyGradient.new(environment, model, optimizer, discount_factor: discount_factor)

# Train the agent (one policy update at the end of each episode)
agent.train(episodes: episodes, steps_per_episode: max_actions_per_episode)

rewards = agent.logs[:episode_reward]
window = [100, rewards.size].min
puts "Mean reward over the last #{window} episodes: #{rewards.last(window).sum / window}"

# Save the trained policy and load it back, as a separate run would
Dir.mktmpdir do |dir|
  path = File.join(dir, "monte_carlo_policy_gradient_gridworld.pth")
  agent.save(path)
  agent.load(path)
end

# Run the learned policy
puts "Learned Policy"
2.times do |i|
  warn "Starting episode #{i}"
  state = environment.reset
  max_actions_per_episode.times do
    action = agent.predict(state)
    state, _, done = environment.step(action)
    # warn "State: #{state} Action: #{action} Done: #{done}"
    if done
      puts "Goal reached!"
      break
    end
  end
end
