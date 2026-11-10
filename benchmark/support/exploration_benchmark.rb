# frozen_string_literal: true

require "reinforce"

# Measures how well an agent explores: it acts in a StateCoverage-wrapped
# environment for a fixed budget of steps, under several fixed seeds, and
# reports the fraction of reachable states it visited and how many of its
# episodes reached the goal.
module ExplorationBenchmark
  # coverage: fraction of reachable states visited within the budget.
  # goal_episodes: episodes that ended by reaching the goal (done).
  Result = Data.define(:seed, :coverage, :goal_episodes, :seconds) do
    def to_s
      format("seed=%d coverage=%.2f goal-episodes=%d time=%.1fs", seed, coverage, goal_episodes, seconds)
    end
  end

  module_function

  # @parameter environment [#call] builds a fresh environment (after seeding).
  # @parameter agent [#call, nil] builds an agent for the wrapped
  #   environment, trained for the budget via #train; nil means a uniformly
  #   random policy.
  # @parameter steps [Integer] environment steps per seed.
  # @parameter steps_per_episode [Integer]
  def run(seeds:, environment:, steps:, steps_per_episode:, agent: nil)
    seeds.map do |seed|
      srand(seed)
      Torch.manual_seed(seed) if defined?(Torch)
      env = Reinforce::Wrappers::StateCoverage.new(environment.call)
      goal_episodes = 0

      started = Process.clock_gettime(Process::CLOCK_MONOTONIC)
      if agent
        learner = agent.call(env)
        learner.train(episodes: steps / steps_per_episode, steps_per_episode:)
        goal_episodes = learner.metrics[:episode_reward].count(&:positive?)
      else
        (steps / steps_per_episode).times do
          env.reset
          steps_per_episode.times do
            _state, _reward, done = env.step(env.actions.sample)
            if done
              goal_episodes += 1
              break
            end
          end
        end
      end
      env.reset # closes the last episode's coverage record
      seconds = Process.clock_gettime(Process::CLOCK_MONOTONIC) - started

      Result.new(seed:, coverage: env.coverage, goal_episodes:, seconds:)
    end
  end

  def report(name, results)
    mean = results.sum(&:coverage) / results.size
    lines = results.map { |result| "  #{result}" }
    "#{name}: mean coverage=#{format("%.2f", mean)}\n#{lines.join("\n")}"
  end
end
