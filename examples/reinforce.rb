# frozen_string_literal: true

require 'reinforce/algorithms/reinforce'
require 'reinforce/environment/gridworld'
require 'torch'

# Create the environment
size = 50
start = [0, 0]
goal = [size - 1, size - 1]
obstacles = Array.new(10) { |_| [1 + rand(size - 2), 1 + rand(size - 2)] }
environment = Reinforce::Environment::Gridworld.new(size, start, goal, obstacles)
state_size = environment.state_size
num_actions = environment.actions.size

# Parameters
learning_rate = 0.01
discount_factor = 0.99
episodes = 1000

# input to the network is the current state
# output of the network is the log probabilities of each action
class NNPolicy
  def_delegate :forward, to: :@architecture

  def initialize(state_size, num_actions)
    @architecture = Torch::NN::Sequential.new(
      Torch::NN::Linear.new(state_size, 512),
      Torch::NN::ReLU.new,
      Torch::NN::Linear.new(512, 512),
      Torch::NN::ReLU.new,
      Torch::NN::Linear.new(512, num_actions)
    )
    @architecture.train # Enable training mode
    @log_probs = []
    @rewards = []
  end

  def reset
    @log_probs = []
    @rewards = []
  end
end

# Create the policy model
model = NNPolicy.new(state_size, num_actions)

# Create the optimizer
optimizer = Torch::Optim::Adam.new(model.parameters, lr: learning_rate)

# Create the agent
agent = Reinforce::Algorithms::Reinforce.new(state_size, num_actions, discount_factor, model, optimizer)

# Training loop
episodes.times do
  state = environment.reset
  # trajectory = []
  # actions = []
  # rewards = []

  # Episode loop
  loop do
    # Choose an action
    action = agent.choose_action(state)

    # Take the action and observe the next state and reward
    next_state, reward, done = environment.step(action)

    agent.get_reward(reward)

    # Collect the current state, action, and reward
    # trajectory << state
    # actions << action
    # rewards << reward

    # Update the current state
    state = next_state

    break if done # Reached the goal state
  end

  # Update policy at the end of each episode
  agent.update_policy
  # # Update the model based on the collected states, actions, and rewards
  # # agent.update(states, actions, rewards)
end

# Print the learned policy
puts 'Learned Policy'
(0...num_states).each do |state|
  action = agent.choose_action(state)
  puts "State #{state}: Action #{action}"
end
