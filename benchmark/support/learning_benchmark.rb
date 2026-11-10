# frozen_string_literal: true

require "reinforce"

# Trains an algorithm from scratch under several fixed seeds and evaluates
# the resulting greedy policy, so a benchmark can assert that the algorithm
# still learns -- rather than only that its code runs, which is what the
# unit tests in test/ check.
module LearningBenchmark
  # The outcome of training and evaluating one seed.
  #
  # success_rate: fraction of evaluation episodes that reached a terminal
  #   (goal) state within `max_steps`.
  # mean_return / mean_steps: averaged over the evaluation episodes.
  # seconds: wall-clock training time.
  Result = Data.define(:seed, :success_rate, :mean_return, :mean_steps, :seconds) do
    def to_s
      format("seed=%d success=%.2f return=%.2f steps=%.1f time=%.1fs",
        seed, success_rate, mean_return, mean_steps, seconds)
    end
  end

  module_function

  # @parameter seeds [Array(Integer)]
  # @parameter environment [#call] builds a fresh environment; it is called
  #   after seeding, so randomized layouts (e.g. GridWorld obstacles) are
  #   reproducible per seed.
  # @parameter agent [#call] builds an agent for the given environment.
  # @parameter train [Hash] keyword arguments passed to the agent's #train.
  # @parameter eval_episodes [Integer] greedy evaluation episodes per seed.
  # @parameter max_steps [Integer] step limit per evaluation episode.
  # @returns [Array(Result)]
  def run(seeds:, environment:, agent:, train:, eval_episodes:, max_steps:)
    seeds.map do |seed|
      srand(seed)
      Torch.manual_seed(seed)
      env = environment.call
      learner = agent.call(env)

      started = Process.clock_gettime(Process::CLOCK_MONOTONIC)
      learner.train(**train)
      seconds = Process.clock_gettime(Process::CLOCK_MONOTONIC) - started

      learner.eval if learner.respond_to?(:eval)
      evaluation = evaluate(learner, env, episodes: eval_episodes, max_steps: max_steps)
      Result.new(seed: seed, seconds: seconds, **evaluation)
    end
  end

  # Run the agent's greedy policy (`predict(state)`, deterministic by
  # default) and report how often it reaches a terminal state.
  def evaluate(agent, environment, episodes:, max_steps:)
    outcomes = Array.new(episodes) do
      state = environment.reset
      total = 0.0
      steps = 0
      done = false
      while steps < max_steps && !done
        state, reward, done = environment.step(agent.predict(state))
        total += reward
        steps += 1
      end
      [done, total, steps]
    end

    {
      success_rate: outcomes.count(&:first).fdiv(episodes),
      mean_return: outcomes.sum { |outcome| outcome[1] } / episodes,
      mean_steps: outcomes.sum { |outcome| outcome[2] }.fdiv(episodes)
    }
  end

  def mean_success_rate(results)
    results.sum(&:success_rate) / results.size
  end

  def report(name, results)
    lines = results.map { |result| "  #{result}" }
    "#{name}: mean success=#{format("%.2f", mean_success_rate(results))}\n#{lines.join("\n")}"
  end
end
