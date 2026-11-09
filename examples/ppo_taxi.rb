#!/usr/bin/env ruby
# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi.

require "torch"
require "reinforce"
require "forwardable"
require "unicode_plot"

# Why these settings
#
# With Taxi's Gym-standard penalty of -10 for a wrong pickup/dropoff, PPO
# learns within its first few hundred episodes never to press pickup or
# dropoff at all: every step then costs only -1, and the frequent -10s teach
# that avoidance long before the rare +20 dropoff teaches when pressing pays
# off. The agent then never completes the task.
#
# A sweep (seeds 1-2, 1000 rollouts each) over the penalty (-10, -2, -1) and
# PPO's entropy coefficient (0.01, 0.05) found exactly one combination that
# learns: a penalty of -1 (no extra cost beyond the ordinary step cost) with
# an entropy coefficient of 0.05, which keeps the policy exploring long
# enough to find the dropoff reward. It solved the task optimally on both
# seeds -- 8 steps, the shortest possible route -- while every other
# combination never completed a single episode. Both changes are needed:
# -1 with the default 0.01 still fails, as does -2 or -10 with 0.05.
illegal_action_reward = -1
entropy_coefficient = 0.05

# Create the environment
environment = Reinforce::Environments::Taxi.new(
  randomize: false,
  observe_passenger_and_destination: true,
  dropoff_reward: 20,
  illegal_action_reward: illegal_action_reward
)

# Parameters
learning_rate = 2.5e-4
# PPO's training "episodes" are fixed-length rollouts of
# max_actions_per_episode steps, each spanning several actual Taxi episodes
# once the agent learns. 1000 rollouts is enough to converge with the
# settings above (it used to be 2500, with settings that never learned).
episodes = 1_000
max_actions_per_episode = 125

# Create the agent
agent = Reinforce::Algorithms::PPO.new(environment, learning_rate, entropy_coefficient: entropy_coefficient)
puts "State size: #{environment.state_size}"
# Train the agent
agent.train(episodes:, steps_per_episode: max_actions_per_episode, progress_output: $stdout)

agent.save("taxi_ppo.pth")

plot = UnicodePlot.lineplot(Reinforce.moving_average(agent.logs[:loss], 25), title: "Loss", width: 100, height: 20)
plot.render
plot = UnicodePlot.lineplot(Reinforce.moving_average(agent.logs[:episode_reward], 25), title: "Rewards", width: 100, height: 20)
plot.render
plot = UnicodePlot.lineplot(Reinforce.moving_average(agent.logs[:episode_length], 25), title: "Episode Length", width: 100, height: 20)
plot.render

# put in eval mode
agent.eval
# Print the learned policy
puts "Learned Policy"

10.times do
  state = environment.reset
  max_actions_per_episode.times do
    action = agent.predict(state)
    # warn "action: #{action}"
    state, _, done = environment.step(action)
    # warn "State: #{state}, Action: #{action}"
    # environment.render
    if done
      warn "Goal reached!"
      break
    end
  end
end
