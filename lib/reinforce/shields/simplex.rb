# frozen_string_literal: true

require "torch"

module Reinforce
  module Shields
    # The probability simplex: vectors of non-negative components summing
    # to 1, e.g. the shares of an allocation.
    module Simplex
      module_function

      # The Euclidean projection of each row of `values` onto the simplex
      # (Held et al., 1974; Duchi et al., 2008): the closest vector of
      # non-negative shares summing to 1. Unlike normalizing, it can set
      # small components to exactly 0. Differentiable almost everywhere.
      #
      # @parameter values [Torch::Tensor] a batch x components tensor.
      # @returns [Torch::Tensor] the projections, shaped like `values`.
      def project(values)
        components = values.shape[1]
        sorted, _indices = Torch.sort(values, 1, true)
        excess = Torch.cumsum(sorted, 1) - 1.0
        ranks = Torch.arange(1, components + 1, dtype: values.dtype).reshape(1, components)
        # The support is the largest k with sorted[k] > (sum of the k largest - 1) / k.
        support = (sorted - (excess / ranks)).gt(0.0).sum(1, keepdim: true)
        threshold = excess.gather(1, support - 1) / support.to(values.dtype)
        Torch.clamp(values - threshold, min: 0.0)
      end
    end
  end
end
