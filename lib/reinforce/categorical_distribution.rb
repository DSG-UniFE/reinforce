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
    @logits = logits.dup.freeze
  end

  # def probabilities
  #   @probabilities ||= @logits.map { |logit| sigmoid(logit) }.freeze
  # end

  def softmax(logits)
    max_logit = logits.max
    exps = logits.map { |logit| Math.exp(logit - max_logit) }
    sum_of_exps = exps.sum
    exps.map { |exp| exp / sum_of_exps }
  end

  def log_probability(index)
    if index.is_a? Integer
      #warn "index is integer: #{index}"
      probs = softmax(@logits)
      Math.log(probs[index])
    else
      logprobs = []
      index = index.to_a
      index.each_with_index do |e, i|
        #warn "index is: #{i} e: #{e} #{@logits[i.to_i][e.to_i]} #{Math.log(sigmoid(@logits[i.to_i][e.to_i]))}"
        probs = softmax(@logits[i.to_i])
        logprobs << Math.log(probs[e.to_i])
      end
      logprobs
    end
  end

  def sample
    # In order not to leave the log probability space, we sample using the Gumbel-max trick.
    # See https://en.wikipedia.org/wiki/Categorical_distribution#Sampling_via_the_Gumbel_distribution and
    # https://stats.stackexchange.com/questions/64081/how-do-i-sample-from-a-discrete-categorical-distribution-in-log-space
    x = @logits.map { |logit| logit - Math.log(-Math.log(rand)) }
    argmax(x)
  end

  def greedy
    warn "logits: #{@logits}"
    # return the index of the action with the highest logit (equivalent to the
    # action with the highest probability)
    argmax(@logits)
  end

  def entropy
    # The entropy of a categorical distribution is given by:
    # H(p) = - sum_i p_i log(p_i)
    # where p_i is the probability of the i-th action.
    # We can compute the entropy in the logit space as:
    # H(p) = - sum_i p_i log(sigmoid(logits_i))
    #      = - sum_i p_i log(1 / (1 + exp(-logits_i)))
    #      = - sum_i p_i (-logits_i + log(1 + exp(-logits_i)))
    #      = sum_i p_i logits_i - p_i log(1 + exp(logits_i))
    #      = sum_i p_i logits_i - p_i (log(1 + exp(logits_i)) - logits_i)
    #      = sum_i p_i logits_i + p_i logits_i - p_i log(1 + exp(logits_i))
    #      = sum_i p_i logits_i + p_i logits_i - p_i log(1 + exp(logits_i))
    @logits.each.map do |logit|
      if logit.is_a? Array
          logit.each.map do |lt|
          pi = sigmoid(lt)
          pi * lt + pi * lt - pi * Math.log(1 + Math.exp(lt))
          end.sum
      else
        p_i = sigmoid(logit)
        p_i * logit + p_i * logit - p_i * Math.log(1 + Math.exp(logit))
      end
    end.sum
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
