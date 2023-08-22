# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi.

require 'forwardable'

class CategoricalDistribution
  extend Forwardable

  def_delegators :@logits, :size

  def initialize(logits:)
    @logits = logits.dup.freeze
  end

  # def probabilities
  #   @probabilities ||= @logits.map { |logit| sigmoid(logit) }.freeze
  # end

  def log_probability(index)
    Math.log(sigmoid(@logits[index]))
  end

  def sample
    # In order not to leave the log probability space, we sample using the Gumbel-max trick.
    # See https://en.wikipedia.org/wiki/Categorical_distribution#Sampling_via_the_Gumbel_distribution and
    # https://stats.stackexchange.com/questions/64081/how-do-i-sample-from-a-discrete-categorical-distribution-in-log-space
    x = @logits.map { |logit| logit - Math.log(-Math.log(rand)) }
    argmax(x)
  end

  def greedy
    # return the index of the action with the highest logit (equivalent to the
    # action with the highest probability)
    argmax(@logits)
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
