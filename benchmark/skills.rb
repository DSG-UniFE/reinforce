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
# attributes its final state to the right skill. Measured 2026-11-11:
#
#   training             identified   distinct greedy end cells
#   none                 0.12-0.17    2-4 / 8   (chance: 0.125)
#   1,000 rollouts       0.83-0.88    7 / 8
#   3,000 rollouts       0.88-1.00    7-8 / 8
#
# Known limitation, not checked here: on four-rooms GridWorlds DIAYN's
# skills are perfectly distinguishable but nearly all stay in the starting
# room -- its objective rewards being distinguishable, not going far (the
# well-known DIAYN failure that LSD/DADS and exploration bonuses address).
describe "skill discovery benchmark" do
  def open_grid
    -> { Reinforce::Environments::GridWorld.new(10, [0, 0], [9, 9], []) }
  end

  def diayn
    lambda do |environment|
      Reinforce::Algorithms::DIAYN.new(
        environment, skills: 8, agent: ->(env) { Reinforce::Algorithms::PPO.new(env, learning_rate: 1e-3) }
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
