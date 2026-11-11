# frozen_string_literal: true

require "reinforce"

# Measures how distinct the skills learned by an unsupervised skill
# discovery method (DIAYN) are: each skill is run from the environment's
# start, and its final state is recorded.
module SkillBenchmark
  # distinct_ends: distinct final cells of the skills' greedy runs.
  # identified: fraction of runs (greedy and sampled) whose final state the
  #   discriminator attributes to the skill that produced it (chance level:
  #   1 / skills).
  Result = Data.define(:seed, :skills, :distinct_ends, :identified, :seconds) do
    def to_s
      format("seed=%d distinct-greedy-ends=%d/%d identified=%.2f time=%.1fs", seed, distinct_ends, skills, identified, seconds)
    end
  end

  module_function

  # @parameter environment [#call] builds a fresh GridWorld-like
  #   environment (with a #state position reader).
  # @parameter diayn [#call] builds a DIAYN for it.
  # @parameter train [Hash] keyword arguments for DIAYN#train (nil: untrained).
  def run(seeds:, environment:, diayn:, train:, steps:, sampled_runs: 5)
    seeds.map do |seed|
      srand(seed)
      Torch.manual_seed(seed)
      learner = diayn.call(environment.call)
      started = Process.clock_gettime(Process::CLOCK_MONOTONIC)
      learner.train(**train) if train
      seconds = Process.clock_gettime(Process::CLOCK_MONOTONIC) - started

      evaluation = environment.call
      skills = learner.config.skills
      greedy_ends = []
      identified = 0
      runs = 0
      skills.times do |skill|
        [[true, 1], [false, sampled_runs]].each do |deterministic, count|
          count.times do
            state = evaluation.reset
            steps.times { state, = evaluation.step(learner.predict(state, skill:, deterministic:)) }
            greedy_ends << evaluation.state.dup if deterministic
            probabilities = learner.skill_probabilities(state)
            identified += 1 if probabilities.index(probabilities.max) == skill
            runs += 1
          end
        end
      end

      Result.new(seed:, skills:, distinct_ends: greedy_ends.uniq.size, identified: identified.fdiv(runs), seconds:)
    end
  end

  def report(name, results)
    lines = results.map { |result| "  #{result}" }
    "#{name}: mean identified=#{format("%.2f", results.sum(&:identified) / results.size)}\n#{lines.join("\n")}"
  end
end
