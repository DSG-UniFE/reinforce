# frozen_string_literal: true

class CategoricalDistribution
  def_delegate :size, to: :@log_probs

  def initialize(log_probs:)
    @log_probs = log_probs.dup.freeze
  end

  def probabilities
    @probabilities ||= @log_probs.map { |log_prob| Math.exp(log_prob) }.freeze
  end

  def log_probability(index)
    @log_probs[index]
  end

  def sample
    # In order not to leave the log probability space, we sample using the Gumbel-max trick.
    # See https://en.wikipedia.org/wiki/Categorical_distribution#Sampling_via_the_Gumbel_distribution and
    # https://stats.stackexchange.com/questions/64081/how-do-i-sample-from-a-discrete-categorical-distribution-in-log-space
    x = @log_probs.map { |log_prob| log_prob - Math.log(-Math.log(rand)) }
    argmax(x)
  end

  private

  def argmax(array)
    argmax = 0
    max = array[0]
    array.each_with_index do |value, index|
      if value > max
        max = value
        argmax = index
      end
    end
    argmax
  end
end
