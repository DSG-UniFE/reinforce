# frozen_string_literal: true

require "ltn"

module Reinforce
  module Shields
    # The Łukasiewicz existential quantifier: min(1, sum of the truth
    # values), the aggregation of Łukasiewicz's strong disjunction.
    #
    # It is exact for the probability of a disjunction of mutually exclusive
    # events, whose probabilities sum to at most 1: e.g. "the destination
    # chosen for an allocation is safe", over the destinations, where each
    # term is P(destination chosen and safe).
    class BoundedSum < LTN::FuzzyOps::AggregationOperator
      def call(xs, dim: nil, keepdim: false, mask: nil)
        unless mask.nil?
          LTN::FuzzyOps.check_mask(mask, xs)
          # 0 is the neutral element of the sum, so masked-out values add nothing.
          xs = LTN::FuzzyOps.where(mask, xs, Torch.zeros_like(xs))
        end
        Torch.clamp_max(Torch.sum(xs, dim: LTN::FuzzyOps.reduction_dims(dim), keepdim: keepdim), 1.0)
      end
    end
  end
end
