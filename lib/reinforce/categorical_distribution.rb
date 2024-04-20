# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi.

require 'forwardable'

class CategoricalDistributionFactory
  def self.create_from_logits(logits)
    CategoricalDistribution.new(logits:)
  end

  def self.create_from_frequencies(frequencies)
    CategoricalDistributionFrequency.new(frequencies:)
  end
end

class CategoricalDistributionFrequency
  def initialize(frequencies:)
    @frequencies = frequencies
    @cumulative_probabilities = compute_cumulative_probabilities(frequencies)
  end

  def sample
    random_value = rand
    @cumulative_probabilities.each_with_index do |cumulative_prob, index|
      return index if random_value < cumulative_prob
    end
    # Return the last index if the random value is greater than or equal to 1
    @cumulative_probabilities.length - 1
  end

  def greedy
    # Return the index with the highest frequency as the greedy choice
    max_index = @frequencies.index(@frequencies.max)
    max_index.nil? ? sample : max_index
  end

  private

  def compute_cumulative_probabilities(frequencies)
    total_frequency = frequencies.sum.to_f
    cumulative_probabilities = [frequencies[0] / total_frequency]

    1.upto(frequencies.length - 1).each do |i|
      cumulative_probabilities[i] = cumulative_probabilities[i - 1] + frequencies[i] / total_frequency
    end

    cumulative_probabilities
  end
end

class CategoricalDistribution
  extend Forwardable

  def_delegators :@logits, :size

  def initialize(logits:)
    @logits = logits
  end

  # def probabilities
  #   @probabilities ||= @logits.map { |logit| sigmoid(logit) }.freeze
  # end

  def log_probability(index)
    prbs = Torch.sigmoid(@logits)
    unless prbs.size.to_a.length == 1
      prbs.log[Torch.arange(prbs.size(0)), index.long]
    else
      prbs.log[Torch.tensor(index).long]
    end
  end

  def sample
    # In order not to leave the log probability space, we sample using the Gumbel-max trick.
    # See https://en.wikipedia.org/wiki/Categorical_distribution#Sampling_via_the_Gumbel_distribution and
    # https://stats.stackexchange.com/questions/64081/how-do-i-sample-from-a-discrete-categorical-distribution-in-log-space
    #x = @logits.map { |logit| logit - Math.log(-Math.log(rand)) }
    # Use torch instead of Math
    x = @logits - Torch.log(-Torch.log(Torch.rand_like(@logits)))
    x.argmax
  end

  def mode
    # return the index of the action with the highest probability
    @logits.argmax
  end

  def greedy
    # return the index of the action with the highest logit (equivalent to the
    # action with the highest probability)
    @logits.argmax
  end

  def entropy
    # The entropy of a categorical distribution is given by:
    # H(p) = - \sum_i p_i \log(p_i)
    # where p_i is the probability of the i-th action.
    # p_i are calculated using the sigmoid function instead of the softmax.
    - Torch.sum(Torch.softmax(@logits, dim: 0) * Torch.log_softmax(@logits, dim: 0))
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

  # This function converts from a logit to the corresponding probability.
  def sigmoid(logit)
    1 / (1 + Math.exp(-logit))
  end
end
