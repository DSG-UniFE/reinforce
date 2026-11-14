# frozen_string_literal: true

require "ltn"

module Reinforce
  module Shields
    # The product universal quantifier: the product of the truth values,
    # the aggregation of the product t-norm.
    #
    # It is exact for the probability of a conjunction of independent
    # events: e.g. "every workflow meets its SLO", over the workflows, when
    # their violations are taken as independent. (The counterpart of
    # BoundedSum for the existential.)
    class Product < LTN::FuzzyOps::AggregationOperator
      def call(xs, dim: nil, keepdim: false, mask: nil)
        unless mask.nil?
          LTN::FuzzyOps.check_mask(mask, xs)
          # 1 is the neutral element of the product, so masked-out values change nothing.
          xs = LTN::FuzzyOps.where(mask, xs, Torch.ones_like(xs))
        end
        return Torch.prod(xs) if dim.nil? && !keepdim

        dims = if dim.nil?
          (0...xs.dim).to_a
        else
          Array(dim).map { |d| d.negative? ? d + xs.dim : d }
        end
        dims.sort.reverse.reduce(xs) { |product, d| Torch.prod(product, d, keepdim: keepdim) }
      end
    end
  end
end
