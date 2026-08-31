# frozen_string_literal: true

require_relative "../support/torch_test_helpers"
require "reinforce/categorical_distribution"

# Regression tests for the CategoricalDistribution#log_probability and
# #entropy bug: they used to derive per-action probabilities from
# Torch.sigmoid(logits) instead of Torch.softmax(logits), which only
# happens to be correct for exactly two actions with symmetric logits.
# These tests pin down the fix against action spaces with more than two
# actions (the common case in this library: GridWorld has 4, Taxi has 6),
# and against batched logits (shape [batch, num_actions]), which is how
# PPO and REINFORCE call this class during training.
describe CategoricalDistribution do
  it "produces per-action probabilities (from log_probability) that sum to 1 across 3 actions" do
    TorchTestHelpers.with_torch_seed do
      logits = Torch.tensor([0.5, -1.0, 2.0], dtype: :float32)
      distribution = CategoricalDistribution.new(logits: logits)

      total_probability = (0...3).sum do |action|
        Math.exp(distribution.log_probability(action).item)
      end

      expect(total_probability).to be_within(1e-4).of(1.0)
    end
  end

  it "reports the closed-form entropy (ln 3) for 3 equally likely actions" do
    # With the old sigmoid-based formula this returned ~1.648 instead of
    # the correct ln(3) ~= 1.0986: sigmoid(0) == 0.5 for every action, so
    # the "probabilities" summed to 1.5 instead of 1.
    logits = Torch.tensor([0.0, 0.0, 0.0], dtype: :float32)
    distribution = CategoricalDistribution.new(logits: logits)

    expect(distribution.entropy.item).to be_within(1e-4).of(Math.log(3))
  end

  it "reports near-zero entropy for an almost-deterministic distribution" do
    logits = Torch.tensor([20.0, 0.0, 0.0], dtype: :float32)
    distribution = CategoricalDistribution.new(logits: logits)

    expect(distribution.entropy.item).to be_within(1e-3).of(0.0)
  end

  it "computes entropy per-row for batched logits, not reduced across the batch" do
    # Regression test for the accompanying dim: 0 -> dim: -1 fix: with
    # dim: 0, softmax/log_softmax would normalize across the batch
    # dimension instead of the action dimension whenever logits is
    # batched (shape [batch, num_actions]), which is how PPO's minibatch
    # updates call this class.
    batched_logits = Torch.tensor([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]], dtype: :float32)
    distribution = CategoricalDistribution.new(logits: batched_logits)

    row_entropies = distribution.entropy.to_a

    expect(row_entropies[0]).to be_within(1e-4).of(Math.log(3))
    expect(row_entropies[1]).to be_within(1e-3).of(0.0)
  end

  it "computes log_probability per-row for batched logits, matching the unbatched computation" do
    batched_logits = Torch.tensor([[0.5, -1.0, 2.0], [0.5, -1.0, 2.0]], dtype: :float32)
    batched_distribution = CategoricalDistribution.new(logits: batched_logits)
    unbatched_distribution = CategoricalDistribution.new(logits: batched_logits[0])

    actions = Torch.tensor([2, 2])
    batched_log_probs = batched_distribution.log_probability(actions).to_a

    expect(batched_log_probs[0]).to be_within(1e-4).of(unbatched_distribution.log_probability(2).item)
    expect(batched_log_probs[1]).to be_within(1e-4).of(unbatched_distribution.log_probability(2).item)
  end
end
