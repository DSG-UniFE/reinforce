# frozen_string_literal: true

require_relative "support/learning_benchmark"

# Seeded learning regression benchmark: does each algorithm still *learn*?
#
# Each case trains from scratch under several fixed seeds, then runs the
# greedy policy and checks how many seeds reach the goal. Run with:
#
#   bundle exec sus benchmark/learning.rb
#
# It lives outside test/ so the default `bundle exec sus` run stays fast;
# CI runs it as a separate job.
#
# Budgets and thresholds were calibrated on 2026-11-09 (macOS, Apple
# Silicon, one Torch thread). Every case except PPO on GridWorld solved
# all of its seeds there; thresholds allow one seed fewer, because Torch
# results are only reproducible per platform, and CI runs on Linux. PPO on
# GridWorld fails one seed in five at every entropy coefficient tried
# (0.01, 0.03, 0.1), so it runs five seeds and needs three.
#
# If a change makes a case fail, look at the per-seed report it prints
# before touching the threshold: a seed that used to reach the goal in the
# optimal number of steps and no longer does is exactly what this exists
# to catch.
describe "learning regression benchmark" do
  def gridworld(size, obstacles)
    -> { Reinforce::Environments::GridWorld.new(size, [0, 0], [size - 1, size - 1], obstacles) }
  end

  def fixed_taxi
    lambda do
      Reinforce::Environments::Taxi.new(
        randomize: false, observe_passenger_and_destination: true, dropoff_reward: 20, illegal_action_reward: -1
      )
    end
  end

  def small_network(environment)
    Reinforce::Networks.mlp(environment.state_size, environment.actions.size, hidden_size: 64)
  end

  def expect_learning(name, minimum_success_rate:, **options)
    results = LearningBenchmark.run(**options)
    report = LearningBenchmark.report(name, results)
    inform(report)

    expect(LearningBenchmark.mean_success_rate(results)).to be >= minimum_success_rate
  end

  it "DQN solves a 10x10 GridWorld" do
    expect_learning(
      "DQN / GridWorld 10x10",
      seeds: [1, 2, 3],
      environment: gridworld(10, 5),
      agent: ->(environment) { Reinforce::Algorithms::DQN.new(environment, 0.001, 0.99) },
      train: {episodes: 300, steps_per_episode: 150},
      eval_episodes: 1,
      max_steps: 150,
      minimum_success_rate: 2.0 / 3
    )
  end

  it "DQN solves fixed-layout Taxi" do
    expect_learning(
      "DQN / Taxi (fixed)",
      seeds: [1, 2, 3],
      environment: fixed_taxi,
      agent: ->(environment) { Reinforce::Algorithms::DQN.new(environment, 0.001, 0.99) },
      train: {episodes: 300, steps_per_episode: 100},
      eval_episodes: 1,
      max_steps: 100,
      minimum_success_rate: 2.0 / 3
    )
  end

  it "PPO solves fixed-layout Taxi" do
    expect_learning(
      "PPO / Taxi (fixed)",
      seeds: [1, 2, 3],
      environment: fixed_taxi,
      agent: ->(environment) { Reinforce::Algorithms::PPO.new(environment, 2.5e-4, entropy_coefficient: 0.05) },
      train: {episodes: 300, steps_per_episode: 125},
      eval_episodes: 1,
      max_steps: 100,
      minimum_success_rate: 2.0 / 3
    )
  end

  it "PPO solves a 5x5 GridWorld on most seeds" do
    expect_learning(
      "PPO / GridWorld 5x5",
      seeds: [1, 2, 3, 4, 5],
      environment: gridworld(5, 3),
      agent: ->(environment) { Reinforce::Algorithms::PPO.new(environment, 0.001, entropy_coefficient: 0.03) },
      train: {episodes: 300, steps_per_episode: 64},
      eval_episodes: 1,
      max_steps: 50,
      minimum_success_rate: 3.0 / 5
    )
  end

  it "SAC solves a 5x5 GridWorld" do
    expect_learning(
      "SAC / GridWorld 5x5",
      seeds: [1, 2, 3],
      environment: gridworld(5, 3),
      agent: lambda do |environment|
        Reinforce::Algorithms::SAC.new(
          environment,
          policy_model: small_network(environment),
          q1_model: small_network(environment),
          q2_model: small_network(environment),
          q1_target_model: small_network(environment),
          q2_target_model: small_network(environment),
          learning_rate: 1e-3
        )
      end,
      train: {episodes: 200, steps_per_episode: 50},
      eval_episodes: 1,
      max_steps: 50,
      minimum_success_rate: 2.0 / 3
    )
  end
end
