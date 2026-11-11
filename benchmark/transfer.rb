# frozen_string_literal: true

require "reinforce"

# Zero-shot transfer benchmark: after reward-free pretraining, can an agent
# act optimally on objectives it has never been trained on? Run with:
#
#   bundle exec sus benchmark/transfer.rb
#
# Successor features with generalized policy improvement on an 11x11
# MultiGoalGridWorld four rooms (one goal per room; one-hot observations;
# exploring starts during training only). Training sees only transition
# features -- which goal was reached, plus a per-step feature -- for four
# basis tasks ("reach goal k", step cost 0.01), 1,000 episodes of up to 50
# steps. Each test task is then solved zero-shot from the fixed start, and
# its discounted return (gamma 0.95, the agent's own) compared with the
# exact optimum. Measured 2026-11-11, 8 test tasks, seeds 1-5: 8/8 optimal
# on every seed (~75s per seed). Without exploring starts, training almost
# never reaches the farthest goal (once in 50,000 transitions) and every
# task that needs it fails; at 600 episodes, 6-8/8 depending on the seed.
describe "zero-shot transfer benchmark" do
  def step_cost = -0.01

  def environment(task: nil, random_start: false)
    Reinforce::Environments::MultiGoalGridWorld.four_rooms(observation: :one_hot, task:, random_start:)
  end

  def discounted_return(agent, task, max_steps:, discount_factor:)
    env = environment(task:)
    state = env.reset
    total = 0.0
    max_steps.times do |t|
      state, reward, done = env.step(agent.predict(state, task:))
      total += reward * discount_factor**t
      break if done
    end
    total
  end

  it "successor features with GPI solve unseen tasks optimally" do
    test_tasks = {
      "reach goal 3" => [0, 0, 0, 1, step_cost],
      "goal 0 or 1" => [1, 1, 0, 0, step_cost],
      "avoid goal 0, reach goal 1" => [-1, 1, 0, 0, step_cost],
      "far goal worth more" => [0, 0.5, 0.5, 1, step_cost],
      "costly steps" => [0.3, 0, 0, 1, -0.05],
      "avoid every goal" => [-1, -1, -1, -1, 0.0],
      "mixed preferences" => [0.2, -0.5, 0.9, 0.4, step_cost],
      "near goal penalized" => [-0.5, 0.6, 0.7, 0.2, step_cost]
    }
    basis = (0...4).map { |goal| Array.new(4, 0.0).tap { |task| task[goal] = 1.0 } + [step_cost] }

    [1, 2].each do |seed|
      srand(seed)
      Torch.manual_seed(seed)
      agent = Reinforce::Algorithms::SuccessorFeatures.new(environment(random_start: true), tasks: basis)
      started = Process.clock_gettime(Process::CLOCK_MONOTONIC)
      agent.train(episodes: 1000, steps_per_episode: 50)
      seconds = Process.clock_gettime(Process::CLOCK_MONOTONIC) - started

      results = test_tasks.map do |name, task|
        achieved = discounted_return(agent, task, max_steps: 50, discount_factor: agent.config.discount_factor)
        optimal = environment.best_return(task, max_steps: 50, discount_factor: agent.config.discount_factor)
        [name, achieved, optimal]
      end
      report = "SF + GPI / multi-goal four rooms, seed #{seed} (#{seconds.round}s):\n" +
        results.map { |name, achieved, optimal| format("  %-28s achieved=%6.3f optimal=%6.3f", name, achieved, optimal) }.join("\n")
      inform(report)
      warn(report)

      results.each { |_name, achieved, optimal| expect(achieved).to be_within(1e-4).of(optimal) }
    end
  end
end
