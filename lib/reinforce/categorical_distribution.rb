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

  def log_probability(index)
    # A categorical distribution over the actions is defined by a
    # probability vector that sums to 1 across all actions. Softmax is
    # the correct way to turn a vector of unnormalized logits into such
    # a distribution: softmax(logits)_i = exp(logit_i) / sum_j(exp(logit_j)),
    # the unique normalization consistent with the logits that always
    # sums to 1 across the actions, no matter how many there are.
    #
    # Sigmoid, which we used originally here instead of softmax, maps each
    # logit independently to 1 / (1 + exp(-logit)). Those per-action values do
    # not depend on each other and do not sum to 1 once there are more than two
    # actions, so they are not a valid categorical distribution and give the
    # wrong log-probability for any environment with more than two actions.
    #
    # We use log_softmax directly, rather than Torch.softmax(...).log,
    # because it is numerically more stable (it avoids computing exp()
    # and then log() back to back). dim: -1 normalizes over the action
    # dimension whether @logits is a single vector (shape [num_actions])
    # or a batch of them (shape [batch, num_actions]).
    log_probs = Torch.log_softmax(@logits, dim: -1)
    if log_probs.size.to_a.length == 1
      log_probs[Torch.tensor(index).long]
    else
      log_probs[Torch.arange(log_probs.size(0)), index.long]
    end
  end

  def sample
    # In order not to leave the log probability space, we sample using the Gumbel-max trick.
    # See https://en.wikipedia.org/wiki/Categorical_distribution#Sampling_via_the_Gumbel_distribution and
    # https://stats.stackexchange.com/questions/64081/how-do-i-sample-from-a-discrete-categorical-distribution-in-log-space
    x = @logits - Torch.log(-Torch.log(Torch.rand_like(@logits)))
    x.argmax
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
    #
    # Note: p_i must come from softmax, not sigmoid: softmax(logits)_i is the
    # unique probability distribution over the actions implied by the logits
    # (it sums to 1 across actions), while sigmoid treats each logit as an
    # independent Bernoulli probability that does not sum to 1 once there are
    # more than two actions. Using sigmoid here silently produced the wrong
    # entropy for every action space larger than 2.
    #
    # dim: -1 operates on the action dimension in both the unbatched case
    # (@logits has shape [num_actions]) and the batched case (@logits has
    # shape [batch, num_actions]); using dim: 0, as this used to, would
    # incorrectly normalize/reduce across the batch dimension instead of
    # the action dimension whenever @logits is batched.
    log_probs = Torch.log_softmax(@logits, dim: -1)
    probs = log_probs.exp
    -Torch.sum(probs * log_probs, dim: -1)
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
