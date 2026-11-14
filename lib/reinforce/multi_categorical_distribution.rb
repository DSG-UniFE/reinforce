# frozen_string_literal: true

require_relative "categorical_distribution"

# Independent categorical distributions, one per dimension of a
# multi-discrete action space (see Reinforce::Spaces::MultiDiscrete), over
# one logits tensor whose last dimension concatenates the dimensions'
# logits. An action is a vector of indices, one per dimension; its
# log-probability and the entropy are sums over the dimensions.
class MultiCategoricalDistribution
  # @parameter logits [Torch::Tensor] [sizes.sum] or [batch, sizes.sum].
  # @parameter sizes [Array(Integer)] the choices per dimension.
  # @parameter mask [Torch::Tensor, Array, nil] booleans shaped like logits.
  def initialize(logits:, sizes:, mask: nil)
    logits = CategoricalDistribution.masked(logits, mask) unless mask.nil?
    @sizes = sizes
    @parts = logits.split(sizes, -1).map { |part| CategoricalDistribution.new(logits: part) }
  end

  # @parameter actions [Torch::Tensor, Array] [dims], or [batch, dims].
  def log_probability(actions)
    actions = Torch.tensor(actions) unless actions.is_a?(Torch::Tensor)
    columns = (actions.dim == 1) ? actions.to_a.map { |a| a } : nil
    @parts.each_with_index.map do |part, i|
      part.log_probability(columns ? columns[i] : actions[0.., i])
    end.reduce(:+)
  end

  def entropy = @parts.map(&:entropy).reduce(:+)

  # @returns [Torch::Tensor] one index per dimension.
  def sample = Torch.stack(@parts.map(&:sample))

  def greedy = Torch.stack(@parts.map(&:greedy))

  # @returns [Array(Integer)] the greedy or sampled index per dimension.
  def choose(deterministic: true) = (deterministic ? greedy : sample).to_a.map(&:to_i)
end
