#!/usr/bin/env ruby
# frozen_string_literal: true

# Compares Soft Actor-Critic with and without an LTN shield (and the greedy
# best-fit heuristic, with real or perfect telemetry, and random actions) on
# multi-cluster replica allocation:
#
#   bundle exec ruby examples/shielded_sac_multi_cluster.rb --preset hard --clusters 4 --episodes 60
#   bundle exec ruby examples/shielded_sac_multi_cluster.rb --preset diurnal --steps 1200 --episodes 40
#
# Each agent trains on its own environment with the same seed, then all are
# tested on the same test episodes. The report gives, per agent, the
# percentages of requested replicas that were served, rejected and in
# overflow, during training and testing, and the mean reward per step. With a
# workload whose load varies (the diurnal preset), test results are also given
# by the load the arrivals offered: low (below 0.75 of the capacity), mid and
# high (above the capacity).
require "optparse"
require "reinforce/shields"

options = {preset: "hard", clusters: 4, episodes: 60, steps: 500, test_episodes: 10, seeds: [1], agents: %w[sac shielded greedy oracle random]}
OptionParser.new do |parser|
  parser.on("--preset NAME", %w[easy hard diurnal]) { |value| options[:preset] = value }
  parser.on("--clusters N", Integer) { |value| options[:clusters] = value }
  parser.on("--episodes N", Integer, "training episodes") { |value| options[:episodes] = value }
  parser.on("--steps N", Integer, "steps per episode") { |value| options[:steps] = value }
  parser.on("--test-episodes N", Integer) { |value| options[:test_episodes] = value }
  parser.on("--seeds LIST", Array) { |value| options[:seeds] = value.map(&:to_i) }
  parser.on("--agents LIST", Array, "sac, shielded, greedy, oracle, random") { |value| options[:agents] = value }
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
  when "oracle"
    Reinforce::Environments::MultiCluster::GreedyPolicy.new(environment, oracle: true)
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

LOAD_BANDS = {low: 0.0...0.75, mid: 0.75...1.0, high: 1.0..}

# @returns [Hash] the summary over all the test steps (`all`) and by load band.
def test(agent, environment, episodes)
  totals = Hash.new { |hash, key| hash[key] = {metrics: Hash.new(0.0), reward: 0.0, steps: 0} }
  episodes.times do
    state = environment.reset
    done = false
    until done
      # the environment, not the agent, sees the true free resources
      state, step_reward, done, info = environment.step(agent.predict(state))
      metrics = info[:metrics]
      bands = [:all]
      bands << LOAD_BANDS.find { |_band, range| range.cover?(metrics[:offered_load]) }.first if metrics[:offered_load]
      bands.each do |band|
        metrics.each { |key, value| totals[band][:metrics][key] += value }
        totals[band][:reward] += step_reward
        totals[band][:steps] += 1
      end
    end
  end
  totals.transform_values { |band| summarize(band[:metrics], band[:reward], band[:steps]) }
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
    test_environment = build_environment(options)
    # policies without training read the environment they act on (the oracle's true state)
    agent = build_agent(name, test_environment) unless agent.respond_to?(:train)
    results = test(agent, test_environment, options[:test_episodes])
    puts format_row("#{name} test", results[:all]) + format("  (seed %d)", seed)
    LOAD_BANDS.each_key { |band| puts format_row("  #{band}", results[band]) if results[band] }
  end
end
