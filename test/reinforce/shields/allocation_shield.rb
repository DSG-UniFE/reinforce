# frozen_string_literal: true

require "reinforce/shields"
require_relative "../../support/torch_test_helpers"

# A safety model whose score is the share of the first component, and whose
# preference cost is the share of the second.
class FirstShareSafety
  attr_reader :scored

  def initialize
    @scored = 0
  end

  def score(_observations, shares)
    @scored += 1
    shares.select(1, 0)
  end

  def second_share(_observations, shares)
    shares.select(1, 1)
  end
end

describe Reinforce::Shields::Simplex do
  def project(rows)
    subject.project(Torch.tensor(rows, dtype: :float64)).to_a
  end

  def close?(actual, expected)
    actual.flatten.zip(expected.flatten).all? { |a, b| (a - b).abs < 1e-9 }
  end

  it "projects onto non-negative shares summing to 1" do
    TorchTestHelpers.with_torch_seed do
      projected = subject.project(Torch.randn(50, 5, dtype: :float64) * 3)

      expect(projected.ge(0.0).all.item).to be == true
      expect(close?(projected.sum(1).to_a, Array.new(50, 1.0))).to be == true
    end
  end

  it "keeps points of the simplex and zeroes small components" do
    expect(close?(project([[0.2, 0.3, 0.5]]), [[0.2, 0.3, 0.5]])).to be == true
    expect(close?(project([[0.5, 0.5, 0.5]]), [[1.0 / 3, 1.0 / 3, 1.0 / 3]])).to be == true
    expect(close?(project([[1.0, 0.2, -1.0]]), [[0.9, 0.1, 0.0]])).to be == true
    expect(close?(project([[2.0, 0.0, 0.0]]), [[1.0, 0.0, 0.0]])).to be == true
  end

  it "finds the closest point of the simplex" do
    TorchTestHelpers.with_torch_seed do
      point = Torch.randn(1, 3, dtype: :float64)
      distance = ((subject.project(point) - point)**2).sum.item
      # a grid over the simplex
      grid = (0..50).flat_map { |i| (0..(50 - i)).map { |j| [i / 50.0, j / 50.0, (50 - i - j) / 50.0] } }
      nearest = grid.map { |candidate| candidate.each_with_index.sum { |value, index| (value - point[0, index].item)**2 } }.min

      expect(distance).to be <= nearest + 1e-12
    end
  end
end

describe Reinforce::Shields::AllocationShield do
  let(:safety) { FirstShareSafety.new }

  it "keeps safe actions, executing their projection" do
    shield = subject.new(safety, threshold: 0.6)
    executed = shield.call([0.0], [0.6, -1.0, -0.6])

    # (a + 1) / 2 = [0.8, 0, 0.2] is on the simplex already
    expect(executed.zip([0.6, -1.0, -0.6]).all? { |a, b| (a - b).abs < 1e-6 }).to be == true
    expect(shield.last_info).to have_keys(interventions: be == 0, corrections: be == 0, safety: be_within(1e-6).of(0.8))
    expect(safety.scored).to be == 1
  end

  it "searches for a safe action when the base action is unsafe" do
    shield = subject.new(safety, threshold: 0.6, step_size: 0.5)
    executed = shield.call([0.0], [-0.6, 0.6, -1.0])
    shares = executed.map { |value| (value + 1.0) / 2.0 }

    expect(shield.last_info).to have_keys(interventions: be == 1, corrections: be == 1)
    expect(shares[0]).to be >= 0.6
    expect(shield.last_info[:safety]).to be_within(1e-6).of(shares[0])
    expect(shares.sum).to be_within(1e-6).of(1.0)
  end

  it "takes the safest candidate when none reaches the threshold" do
    shield = subject.new(safety, threshold: 0.99, search_steps: 2, step_size: 0.1)
    executed = shield.call([0.0], [-0.8, 0.8, -1.0])
    shares = executed.map { |value| (value + 1.0) / 2.0 }

    expect(shield.last_info).to have_keys(interventions: be == 1, corrections: be == 1)
    expect(shares[0]).to be > 0.1
    expect(shares[0]).to be < 0.99
  end

  it "prefers the cheapest safe candidate" do
    # Once safe, the search lowers the second share (the preference cost).
    shield = subject.new(safety, preference: :second_share, threshold: 0.3, search_steps: 6, step_size: 0.5, preference_weight: 1.0)
    executed = shield.call([0.0], [-0.6, 0.2, -0.6])
    shares = executed.map { |value| (value + 1.0) / 2.0 }

    expect(shares[0]).to be >= 0.3
    expect(shares[1]).to be < 0.6
  end

  it "has a safety loss that is differentiable in the actions" do
    actions = Torch.tensor([[0.0, 0.0, 0.0], [1.0, -1.0, -1.0]]).requires_grad!(true)
    loss = subject.new(safety).safety_loss(Torch.zeros(2, 1), actions)
    loss.backward

    # -log(1/3) and -log(1), averaged
    expect(loss.item).to be_within(1e-5).of(-Math.log(1.0 / 3) / 2)
    expect(actions.grad[0, 0].item).to be < 0.0
  end

  it "floors the safety probability in the loss" do
    actions = Torch.tensor([[-1.0, 1.0, -1.0]])
    loss = subject.new(safety, probability_floor: 1e-8).safety_loss(Torch.zeros(1, 1), actions)

    expect(loss.item).to be_within(1e-3).of(-Math.log(1e-8))
  end
end
