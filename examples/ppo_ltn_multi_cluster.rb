#!/usr/bin/env ruby
# frozen_string_literal: true

# Compares PPO-LTN (PPO with an LTN shield and a Monte Carlo safety loss)
# with PPO, SAC, SAC with the same shield (SAC-LTN) and training-free
# baselines on multi-cluster replica allocation, with the metrics of the
# NeSy-SAC evaluation:
#
#   bundle exec ruby examples/ppo_ltn_multi_cluster.rb --preset hard --clusters 4 --budget 100000
#   bundle exec ruby examples/ppo_ltn_multi_cluster.rb --agents ppo-ltn,ppo-ltn-det --seeds 1,2,3 --json results.jsonl
#
# Agents:
#
# - ppo: PPO with a tanh-squashed Gaussian policy, no shield;
# - ppo-ltn: PPO with Reinforce::Shields::AllocationShield over
#   MultiClusterSafety, correcting actions in training and testing, and a
#   safety loss over --safety-samples sampled actions per state;
# - ppo-ltn-det: the same with the safety loss on the deterministic action
#   (as in NeSy-SAC);
# - sac, sac-ltn: ContinuousSAC without and with the same shield;
# - greedy (best fit on the delayed telemetry), oracle (best fit on the
#   true free resources) and random.
#
# Every learned agent gets the same budget of environment steps, in
# episodes of --horizon steps (SAC) or rollouts of --rollout steps (PPO;
# a multiple of the horizon avoids cutting episodes). Hyperparameters
# follow the design note's Table C for PPO and NeSy-SAC's Table I for SAC
# and the shield. All agents are then tested on the same --test-episodes
# episodes.
#
# Reported, per agent and seed:
#
# - training: the percentages of the requested replicas served, rejected
#   and in overflow over the whole training (every overflow during training
#   is a failed deployment), the shield's interventions per step, and the
#   time per step;
# - testing: the accumulated reward per episode, the percentages served,
#   rejected and in overflow, the mean latency and cost per served replica,
#   the shield's interventions per step, the time per decision, and the
#   calibration of the fit sensors on the deployments the agent made (the
#   expected calibration error and the Brier score; see
#   docs/sensor-model.md).
#
# Presets: easy and hard (the legacy telemetry, delayed by 0..1 decisions),
# diurnal, and monitored (scripted workload with background tenants and
# realistic telemetry; --no-ledger drops the agent's ledger).
#
# With --json, one JSON object per agent and seed is appended to the file,
# so runs in separate processes can be aggregated.
require "json"
require "optparse"
require "reinforce/shields"

AGENTS = %w[ppo ppo-ltn ppo-ltn-det sac sac-ltn greedy oracle random].freeze

options = {preset: "hard", clusters: 4, budget: 100_000, horizon: 1_000, rollout: 2_000, test_episodes: 10, seeds: [1],
           agents: %w[ppo ppo-ltn ppo-ltn-det sac sac-ltn greedy random], safety_samples: 8, threshold: 0.6, json: nil, ledger: true}
OptionParser.new do |parser|
  parser.on("--preset NAME", %w[easy hard diurnal monitored]) { |value| options[:preset] = value }
  parser.on("--no-ledger", "monitored: without the agent's ledger") { options[:ledger] = false }
  parser.on("--clusters N", Integer) { |value| options[:clusters] = value }
  parser.on("--budget STEPS", Integer, "training steps per learned agent") { |value| options[:budget] = value }
  parser.on("--horizon STEPS", Integer, "steps per episode") { |value| options[:horizon] = value }
  parser.on("--rollout STEPS", Integer, "steps per PPO rollout") { |value| options[:rollout] = value }
  parser.on("--test-episodes N", Integer) { |value| options[:test_episodes] = value }
  parser.on("--seeds LIST", Array) { |value| options[:seeds] = value.map(&:to_i) }
  parser.on("--agents LIST", Array, AGENTS.join(", ")) { |value| options[:agents] = value }
  parser.on("--safety-samples M", Integer, "sampled actions per state in PPO-LTN's safety loss") { |value| options[:safety_samples] = value }
  parser.on("--threshold TAU", Float, "the shield's safety threshold") { |value| options[:threshold] = value }
  parser.on("--json PATH", "append the results to this JSON-lines file") { |value| options[:json] = value }
end.parse!
unknown = options[:agents] - AGENTS
abort "unknown agents: #{unknown.join(", ")}" if unknown.any?

def build_environment(options)
  extra = {}
  extra[:telemetry] = Reinforce::Environments::MultiCluster::Telemetry.new(ledger: false) if options[:preset] == "monitored" && !options[:ledger]
  Reinforce::Environments::MultiCluster.public_send(options[:preset], clusters: options[:clusters], horizon: options[:horizon], **extra)
end

def build_shield(environment, options)
  safety = Reinforce::Shields::MultiClusterSafety.new(environment)
  Reinforce::Shields::AllocationShield.new(safety, preference: :latency, threshold: options[:threshold])
end

# Design note, Table C.
PPO_HYPERPARAMETERS = {
  learning_rate: 3e-4, ppo_epochs: 10, minibatch_size: 256, discount_factor: 0.99, gae_lambda: 0.95, clip_param: 0.2,
  entropy_coefficient: 0.0, max_grad_norm: 0.5, target_kl: 0.02, initial_log_std: 0.0, safety_weight: 0.1,
  network: Reinforce::Networks::Config.new(hidden_size: 64, hidden_layers: 2, activation: Torch::NN::ReLU)
}.freeze

# NeSy-SAC, Table I.
SAC_HYPERPARAMETERS = {
  learning_rate: 3e-4, discount_factor: 0.99, batch_size: 256, replay_size: 100_000, tau: 0.005, warmup_steps: 1_000,
  hidden_size: 64, hidden_layers: 2, safety_weight: 0.1
}.freeze

def build_agent(name, environment, options)
  case name
  when "ppo"
    Reinforce::Algorithms::PPO.new(environment, **PPO_HYPERPARAMETERS)
  when "ppo-ltn"
    Reinforce::Algorithms::PPO.new(environment, shield: build_shield(environment, options), safety_samples: options[:safety_samples],
      **PPO_HYPERPARAMETERS)
  when "ppo-ltn-det"
    Reinforce::Algorithms::PPO.new(environment, shield: build_shield(environment, options), safety_samples: 0, **PPO_HYPERPARAMETERS)
  when "sac"
    Reinforce::Algorithms::ContinuousSAC.new(environment, **SAC_HYPERPARAMETERS)
  when "sac-ltn"
    Reinforce::Algorithms::ContinuousSAC.new(environment, shield: build_shield(environment, options), **SAC_HYPERPARAMETERS)
  when "greedy"
    Reinforce::Environments::MultiCluster::GreedyPolicy.new(environment)
  when "oracle"
    Reinforce::Environments::MultiCluster::GreedyPolicy.new(environment, oracle: true)
  when "random"
    space = environment.action_space
    Object.new.tap { |agent| agent.define_singleton_method(:predict) { |_state| space.sample } }
  end
end

def clock = Process.clock_gettime(Process::CLOCK_MONOTONIC)

def percent(part, whole) = whole.positive? ? 100.0 * part / whole : 0.0

# @returns [Hash] training percentages, interventions per step and ms per step.
def train(agent, name, options)
  episodes, steps = name.start_with?("ppo") ? [options[:budget] / options[:rollout], options[:rollout]] : [options[:budget] / options[:horizon], options[:horizon]]
  started = clock
  agent.train(episodes: episodes, steps_per_episode: steps)
  seconds = clock - started
  metrics = agent.metrics
  total = ->(key) { metrics[key]&.sum.to_f }
  requested = total.call(:episode_requested)
  {served: percent(total.call(:episode_served), requested), rejected: percent(total.call(:episode_rejected), requested),
   overflow: percent(total.call(:episode_overflow), requested), interventions: total.call(:shield_interventions) / (episodes * steps),
   ms_per_step: 1000 * seconds / (episodes * steps)}
end

# @returns [Hash] test metrics over `episodes` episodes of `environment`.
def test(agent, environment, episodes)
  shield = agent.respond_to?(:shield) ? agent.shield : nil
  sensors = Reinforce::Shields::MultiClusterSafety.new(environment)
  calibration = Reinforce::Shields::Calibration.new
  totals = Hash.new(0.0)
  rewards = []
  steps = 0
  interventions = 0
  seconds = 0.0
  episodes.times do
    state = environment.reset
    reward = 0.0
    done = false
    until done
      started = clock
      action = agent.predict(state)
      seconds += clock - started
      interventions += shield.last_info.fetch(:interventions, 0) if shield
      replicas = environment.request[:replicas]
      observation = state
      state, step_reward, done, info = environment.step(action)
      calibrate(calibration, sensors, observation, replicas, info[:deployments], environment.clusters)
      info[:metrics].each { |key, value| totals[key] += value }
      reward += step_reward
      steps += 1
    end
    rewards << reward
  end
  served = totals[:served]
  {reward: rewards.sum / rewards.size, served: percent(served, totals[:requested]), rejected: percent(totals[:rejected], totals[:requested]),
   overflow: percent(totals[:overflow], totals[:requested]), latency: served.positive? ? totals[:latency] / served : 0.0,
   cost: served.positive? ? totals[:cost] / served : 0.0, interventions: interventions.fdiv(steps), ms_per_step: 1000 * seconds / steps,
   ece: calibration.expected_calibration_error, brier: calibration.brier}
end

# Adds the sensors' fit probabilities for the replicas actually sent to
# each cluster, against whether they fitted.
def calibrate(calibration, sensors, observation, replicas, deployments, clusters)
  return if deployments.empty?

  shares = Array.new(clusters + 1, 0.0)
  deployments.each { |deployment| shares[deployment[:cluster]] = deployment[:replicas].to_f / replicas }
  shares[clusters] = 1.0 - shares.sum
  probabilities = Torch.no_grad do
    sensors.fit_probabilities(Torch.tensor([observation], dtype: :float32), Torch.tensor([shares], dtype: :float32))[0].to_a
  end
  deployments.each { |deployment| calibration.add(probabilities[deployment[:cluster]], deployment[:fitted]) }
end

puts format("%s, %d clusters, %d training steps per learned agent, %d test episodes of %d steps", options[:preset], options[:clusters],
  options[:budget], options[:test_episodes], options[:horizon])
options[:seeds].each do |seed|
  options[:agents].each do |name|
    srand(seed)
    Torch.manual_seed(seed)
    agent = build_agent(name, build_environment(options), options)
    training = train(agent, name, options) if agent.respond_to?(:train)
    if training
      puts format("%-12s seed %d  train: served %5.1f%%  rejected %5.1f%%  overflow %5.1f%%  interventions %.2f/step  %.2f ms/step",
        name, seed, *training.values_at(:served, :rejected, :overflow, :interventions, :ms_per_step))
    end

    # the same test episodes for every agent and seed
    srand(10_000)
    test_environment = build_environment(options)
    # policies without training act on the test environment (the oracle reads its true state)
    agent = build_agent(name, test_environment, options) unless agent.respond_to?(:train)
    results = test(agent, test_environment, options[:test_episodes])
    puts format("%-12s seed %d  test:  reward %7.1f  served %5.1f%%  rejected %5.1f%%  overflow %5.1f%%  latency %.3f  cost %.3f  " \
      "interventions %.2f/step  %.2f ms/decision  sensors ECE %.3f Brier %.3f", name, seed,
      *results.values_at(:reward, :served, :rejected, :overflow, :latency, :cost, :interventions, :ms_per_step), results[:ece] || 0.0,
      results[:brier] || 0.0)
    next unless options[:json]

    record = {agent: name, seed: seed, preset: options[:preset], ledger: options[:ledger], clusters: options[:clusters], budget: options[:budget],
              safety_samples: options[:safety_samples], threshold: options[:threshold], train: training, test: results}
    File.open(options[:json], "a") { |file| file.puts(JSON.generate(record)) }
  end
end
