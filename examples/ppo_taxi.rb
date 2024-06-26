#!/usr/bin/env ruby
# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi.

require 'torch'
require 'reinforce'
require 'forwardable'
require 'unicode_plot'

# Create the environment
environment = Reinforce::Environments::TaxiV2.new

# Parameters
learning_rate = 2.5e-4
episodes = 2_500
max_actions_per_episode = 125

# Create the agent
agent = Reinforce::Algorithms::PPO.new(environment, learning_rate)
puts "State size: #{environment.state_size}"
# Train the agent
agent.train(episodes, max_actions_per_episode)


agent.save('taxi_ppo.pth')

plot = UnicodePlot.lineplot(Reinforce.moving_average(agent.logs[:loss], 25), title: "Loss", width: 100, height: 20)
plot.render
plot = UnicodePlot.lineplot(Reinforce.moving_average(agent.logs[:episode_reward], 25), title: "Rewards", width: 100, height: 20)
plot.render
plot = UnicodePlot.lineplot(Reinforce.moving_average(agent.logs[:episode_length], 25), title: "Episode Length", width: 100, height: 20)
plot.render


# put in eval mode
agent.eval
# Print the learned policy
puts 'Learned Policy'

10.times do
  state = environment.reset
  max_actions_per_episode.times do
    action = agent.predict(state)
    #warn "action: #{action} #{action.to_i}"
    state, _, done = environment.step(action.to_i)
    #warn "State: #{state}, Action: #{environment.actions[action]}"
    #environment.render
    if done
      warn 'Goal reached!'
      break
    end 
  end
end
