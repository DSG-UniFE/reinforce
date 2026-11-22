# frozen_string_literal: true

require_relative "support/skill_benchmark"

# Unsupervised skill discovery benchmark: does DIAYN learn skills that are
# genuinely different from each other, with no reward at all? Run with:
#
#   bundle exec sus benchmark/skills.rb
#
# 8 skills on an open 10x10 GridWorld (coordinate observations, 50-step
# episodes, PPO). Each skill is run from the start once greedily and five
# times sampled; a run counts as identified when the discriminator
# attributes its final state to the right skill. The PPO agent uses an
# entropy bonus of 0.1: with the default 0.01, skills partly collapse on
# some seeds (2 of 6 locally, and seed 1 on CI's runners, at 0.60-0.62
# identified), so the outcome depended on the machine. Measured 2026-11-23
# (one LibTorch thread, as in CI), 1,000 rollouts:
#
#   training                         identified   distinct greedy end cells
#   none                             0.12-0.17    2-4 / 8   (chance: 0.125)
#   entropy 0.01 (seeds 1-6)         0.62-0.88    6-8 / 8
#   entropy 0.1 (seeds 1-12)         0.83-1.00    7-8 / 8
#
# Known limitation, not checked here: on four-rooms GridWorlds DIAYN's
# skills are perfectly distinguishable but nearly all stay in the starting
# room -- its objective rewards being distinguishable, not going far (the
# well-known DIAYN failure that LSD/DADS address). Rooms reached by 4
# skills' greedy runs on an 11x11 four rooms (2,000 rollouts):
#
#   DIAYN                                1-2 / 4
#   + count bonus                        1-2 / 4   (explores more in training,
#                                                   still ends near the start)
#   + room-only discriminator features   1, 2, 3 / 4  (can't find other rooms)
#   + both                               4, 3, 2, 2, 3 / 4  (seeds 1-5)
#
# Features define what "different" means and the bonus finds where to be
# different; together they help, but not yet reliably enough to check here.
describe "skill discovery benchmark" do
  def open_grid
    -> { Reinforce::Environments::GridWorld.new(10, [0, 0], [9, 9], []) }
  end

  def diayn
    lambda do |environment|
      Reinforce::Algorithms::DIAYN.new(
        environment, skills: 8, agent: ->(env) { Reinforce::Algorithms::PPO.new(env, learning_rate: 1e-3, entropy_coefficient: 0.1) }
      )
    end
  end

  it "DIAYN learns distinguishable skills without any reward" do
    results = SkillBenchmark.run(seeds: [1, 2, 3], environment: open_grid, diayn:, train: {episodes: 1000, steps_per_episode: 50}, steps: 50)
    report = SkillBenchmark.report("DIAYN / open 10x10, 8 skills", results)
    inform(report)
    warn(report)

    results.each do |result|
      expect(result.identified).to be >= 0.7
      expect(result.distinct_ends).to be >= 6
    end
  end

  it "untrained skills are indistinguishable (the baseline the case above must beat)" do
    results = SkillBenchmark.run(seeds: [1, 2, 3], environment: open_grid, diayn:, train: nil, steps: 50)
    report = SkillBenchmark.report("untrained / open 10x10, 8 skills", results)
    inform(report)
    warn(report)

    results.each { |result| expect(result.identified).to be <= 0.35 }
  end
end
