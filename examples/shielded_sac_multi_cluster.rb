#!/usr/bin/env ruby
# frozen_string_literal: true

# Compares Soft Actor-Critic with and without an LTN shield (and the greedy
# best-fit and random baselines) on multi-cluster replica allocation:
#
#   bundle exec ruby examples/shielded_sac_multi_cluster.rb --preset hard --clusters 4 --episodes 60
#
# Each agent trains on its own environment with the same seed, then all are
# tested on the same test episodes. The report gives, per agent, the
# percentages of requested replicas that were served, rejected and in
# overflow, during training and testing, and the mean reward per step.
require "optparse"
require "reinforce/shields"

options = {preset: "hard", clusters: 4, episodes: 60, steps: 500, test_episodes: 10, seeds: [1], agents: %w[sac shielded greedy random]}
OptionParser.new do |parser|
  parser.on("--preset NAME", %w[easy hard]) { |value| options[:preset] = value }
  parser.on("--clusters N", Integer) { |value| options[:clusters] = value }
  parser.on("--episodes N", Integer, "training episodes") { |value| options[:episodes] = value }
  parser.on("--steps N", Integer, "steps per episode") { |value| options[:steps] = value }
  parser.on("--test-episodes N", Integer) { |value| options[:test_episodes] = value }
  parser.on("--seeds LIST", Array) { |value| options[:seeds] = value.map(&:to_i) }
  parser.on("--agents LIST", Array, "sac, shielded, greedy, random") { |value| options[:agents] = value }
end.parse!

def build_environment(options)
  Reinforce::Environments::MultiCluster.public_send(options[:preset], clusters: options[:clusters], horizon: options[:steps])
end

def build_agent(name, environment)
  case name
  when "sac"
    Reinforce::Algorithms::ContinuousSAC.new(environment, hidden_size: 64)
  when "shielded"
    safety = Reinforce::Shields::MultiClusterSafety.new(environment)
    shield = Reinforce::Shields::AllocationShield.new(safety, preference: :latency)
    Reinforce::Algorithms::ContinuousSAC.new(environment, hidden_size: 64, shield: shield)
  when "greedy"
    Reinforce::Environments::MultiCluster::GreedyPolicy.new(environment)
  when "random"
    space = environment.action_space
    Object.new.tap { |agent| agent.define_singleton_method(:predict) { |_state| space.sample } }
  end
end

# Percentages of the requested replicas, and the reward per step.
def summarize(totals, reward, steps)
  requested = totals[:requested].to_f
  {served: 100 * totals[:served] / requested, rejected: 100 * totals[:rejected] / requested,
   overflow: 100 * totals[:overflow] / requested, reward: reward / steps}
end

def test(agent, environment, episodes)
  totals = Hash.new(0.0)
  reward = 0.0
  steps = 0
  episodes.times do
    state = environment.reset
    done = false
    until done
      state, step_reward, done, info = environment.step(agent.predict(state))
      info[:metrics].each { |key, value| totals[key] += value }
      reward += step_reward
      steps += 1
    end
  end
  summarize(totals, reward, steps)
end

def format_row(label, summary)
  format("%-10s served %5.1f%%  rejected %5.1f%%  overflow %5.1f%%  reward/step %6.3f", label, *summary.values_at(:served, :rejected, :overflow, :reward))
end

puts "#{options[:preset]}, #{options[:clusters]} clusters, #{options[:episodes]} x #{options[:steps]} training steps"
options[:seeds].each do |seed|
  options[:agents].each do |name|
    srand(seed)
    Torch.manual_seed(seed)
    environment = build_environment(options)
    agent = build_agent(name, environment)

    started = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    if agent.respond_to?(:train)
      agent.train(episodes: options[:episodes], steps_per_episode: options[:steps])
      metrics = agent.metrics
      training = %i[requested served rejected overflow].to_h { |key| [key, metrics[:"episode_#{key}"].sum] }
      seconds = Process.clock_gettime(Process::CLOCK_MONOTONIC) - started
      puts format_row("#{name} train", summarize(training, metrics[:episode_reward].sum, options[:episodes] * options[:steps])) +
        format("  (seed %d, %.1f ms/step)", seed, 1000 * seconds / (options[:episodes] * options[:steps]))
    end

    # the same test episodes for every agent
    srand(10_000 + seed)
    puts format_row("#{name} test", test(agent, build_environment(options), options[:test_episodes])) + format("  (seed %d)", seed)
  end
end
