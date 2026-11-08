# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi.

# A categorical distribution over a discrete action space, parametrized by
# unnormalized logits (a [num_actions] tensor, or [batch, num_actions]).
class CategoricalDistribution
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

  # The action-selection rule shared by every algorithm's #predict: the
  # greedy action when deterministic, a sample from the distribution
  # otherwise. Works for both unbatched ([actions]) and single-row batched
  # ([1, actions]) logits, since #greedy/#sample take a flattened argmax.
  #
  # @parameter deterministic [Boolean] greedy if true, sampled if false.
  # @returns [Integer] the chosen action's index.
  def choose(deterministic: true)
    (deterministic ? greedy : sample).item
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
end
