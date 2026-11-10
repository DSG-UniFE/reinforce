# frozen_string_literal: true

module Reinforce
  # Running mean and variance of a stream of scalars, updated in batches
  # with the parallel-variance formula (Chan et al.), as in Gymnasium's and
  # OpenAI Baselines' RunningMeanStd. Used by PPO's reward normalization.
  #
  # The count starts at a tiny `epsilon` (not zero) with mean 0 and
  # variance 1, so the statistics are defined -- and dividing by the
  # standard deviation is safe -- before the first update.
  class RunningMeanStd
    attr_reader :mean, :var, :count

    def initialize(epsilon: 1e-4)
      @mean = 0.0
      @var = 1.0
      @count = epsilon
    end

    # @parameter values [Array(Numeric), Numeric] one batch of samples.
    # @returns [self]
    def update(values)
      values = Array(values).map(&:to_f)
      return self if values.empty?

      batch_count = values.size
      batch_mean = values.sum / batch_count
      batch_var = values.sum { |value| (value - batch_mean)**2 } / batch_count

      delta = batch_mean - @mean
      total_count = @count + batch_count
      @mean += delta * batch_count / total_count
      combined_m2 = @var * @count + batch_var * batch_count + delta**2 * @count * batch_count / total_count
      @var = combined_m2 / total_count
      @count = total_count
      self
    end

    def std
      Math.sqrt(@var)
    end
  end
end
