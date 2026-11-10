# frozen_string_literal: true

require_relative "support/exploration_benchmark"

# Exploration benchmark: how much of a hard-exploration environment does
# each method visit within a fixed budget? Run with:
#
#   bundle exec sus benchmark/exploration.rb
#
# The testbed is a 21x21 four-rooms GridWorld with a sparse reward (+1 only
# at the goal, in the corner opposite the start, two doorways away), 100-step
# episodes and a 30,000-step budget. Measured on 2026-11-10 (seeds 1-3):
#
#   method                     coverage    goal reached in training   greedy solves it
#   random policy              0.42-0.50   never                      -
#   DQN                        0.62-0.83   never                      0/3
#   PPO                        0.37-0.55   34 episodes on one seed    0/3
#   SAC                        0.43-0.56   never                      0/3
#   PPO + count bonus (0.1)    0.86-0.93   0-2 episodes per seed      0/3
#   DQN + count bonus (0.1)    0.68-0.79   never                      0/3
#   SAC + count bonus (0.1)    0.49-0.50   never                      0/3
#
# With one-hot observations (`observation: :one_hot`), same budget:
#
#   method                     coverage    goal reached in training   greedy solves it
#   PPO / DQN / SAC            0.21-0.45 / 0.34-0.55 / 0.27-0.35   never   0/3
#   PPO + count bonus (0.1)    0.95-0.97   3-4 episodes per seed      0/3
#   DQN + count bonus (0.1)    0.75-0.80   0-2 episodes per seed      0/3
#   SAC + count bonus (0.1)    0.86-0.93   2-3 episodes per seed      0/3
#
# One-hot makes every algorithm respond to the bonus (SAC barely does from
# coordinates), but at this budget the goal is still found too rarely to
# learn from. Over 100,000 steps, PPO + count bonus solved one seed of three
# (greedy: optimal 40 steps; stochastic: 19/20). The cases below use
# coordinates, which are much cheaper for SAC (one-hot: ~150s per seed).
#
# The random baseline checks the testbed stays hard enough to tell
# exploration methods apart; each exploration method adds a case that must
# beat it on coverage. Cases judge coverage, not greedy success: on this
# testbed even a policy that reaches the goal hundreds of times in training
# (PPO, 100,000 steps: stochastic policy 11/20 without a bonus, 17/20 with
# one) has a greedy policy that walks into the wall next to a doorway -- a
# state-representation problem (raw coordinates), not an exploration one.
describe "exploration benchmark" do
  def four_rooms
    -> { Reinforce::Environments::GridWorld.four_rooms(size: 21) }
  end

  it "random exploration covers about half of the four-rooms testbed and never reaches the goal" do
    results = ExplorationBenchmark.run(seeds: [1, 2, 3], environment: four_rooms, steps: 30_000, steps_per_episode: 100)
    report = ExplorationBenchmark.report("random policy / four rooms 21x21", results)
    inform(report)
    warn(report)

    results.each do |result|
      expect(result.coverage).to be_within(0.15).of(0.45)
      expect(result.goal_episodes).to be == 0
    end
  end

  it "a count-based bonus makes PPO explore most of the four-rooms testbed" do
    results = ExplorationBenchmark.run(
      seeds: [1, 2, 3], environment: four_rooms, steps: 30_000, steps_per_episode: 100,
      agent: lambda do |environment|
        bonus = Reinforce::IntrinsicRewards::CountBased.new(coefficient: 0.1)
        Reinforce::Algorithms::PPO.new(environment, learning_rate: 1e-3, intrinsic_reward: bonus)
      end
    )
    report = ExplorationBenchmark.report("PPO + count bonus / four rooms 21x21", results)
    inform(report)
    warn(report)

    # Without the bonus PPO covers 0.37-0.55; with it, 0.86-0.93.
    results.each { |result| expect(result.coverage).to be >= 0.75 }
  end
end
