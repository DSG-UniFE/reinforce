# frozen_string_literal: true

require "reinforce/shields"

describe Reinforce::Shields::Product do
  it "multiplies truth values: the probability of independent events all holding" do
    xs = Torch.tensor([[0.9, 0.5], [1.0, 0.2]])
    product = Reinforce::Shields::Product.new
    expect(product.call(xs).item).to be_within(1e-6).of(0.09)
    rows = product.call(xs, dim: 1).to_a
    expect(rows[0]).to be_within(1e-6).of(0.45)
    expect(rows[1]).to be_within(1e-6).of(0.2)
    masked = product.call(xs, dim: 1, mask: Torch.tensor([[true, false], [true, true]]))
    expect(masked[0].item).to be_within(1e-6).of(0.9)
  end

  it "quantifies universally in a Logic Tensor Network" do
    forall = LTN::Quantifier.new(Reinforce::Shields::Product.new, :forall)
    x = LTN::Variable.new("x", Torch.tensor([[0.9], [0.8], [0.5]]))
    holds = LTN::Predicate.new(func: ->(v) { v.squeeze(1) })
    expect(forall.call(x, holds.call(x)).value.item).to be_within(1e-6).of(0.36)
  end
end

# Two dimensions, three options each; option 2 of dimension 0 is unsafe,
# and so is the joint action [1, 1].
class DiscreteShieldTestEnvironment
  attr_reader :taken

  def initialize = @taken = []

  def reset = [0.0]

  def state_size = 1

  def action_mask = [[true, true, true], [true, true, false]]

  def step(action)
    @taken << action
    [[0.0], 0.0, false, {}]
  end
end

class DiscreteShieldTestSafety
  attr_reader :observed

  def initialize = @observed = []

  def option_probabilities(_) = [[1.0, 0.95, 0.2], [0.99, 0.97, 0.99]]

  def probabilities(_, actions)
    actions.map do |a|
      if a == [1, 1] then 0.5
      elsif a[0] == 2 then 0.2
      else 0.99
      end
    end
  end

  def fallbacks(_, action) = [[action[0], 0], [0, 0]]

  def observe(_, action, _) = @observed << action
end

describe Reinforce::Shields::DiscreteShield do
  let(:environment) { DiscreteShieldTestEnvironment.new }
  let(:safety) { DiscreteShieldTestSafety.new }
  let(:shield) { Reinforce::Shields::DiscreteShield.new(environment, safety, threshold: 0.9) }

  it "masks unsafe options, on top of the environment's mask" do
    expect(shield.action_mask).to be == [[true, true, false], [true, true, false]]
  end

  it "keeps the safest allowed option when none is safe enough" do
    safety.define_singleton_method(:option_probabilities) { |_| [[0.1, 0.3, 0.2], [0.99, 0.97, 0.99]] }
    expect(shield.action_mask.first).to be == [false, true, false]
  end

  it "lets safe joint actions through and corrects unsafe ones" do
    shield.step([0, 1])
    expect(shield.last_info).to have_keys(intervened: be == false)
    shield.step([1, 1]) # each option safe, the combination not
    expect(environment.taken).to be == [[0, 1], [1, 0]]
    expect(shield.last_info).to have_keys(intervened: be == true, executed: be == [1, 0])
    expect(shield.interventions).to be == 1
    expect(safety.observed).to be == [[0, 1], [1, 0]]
  end

  it "passes the wrapped environment's other methods through" do
    expect(shield.state_size).to be == 1
    expect(shield.taken).to be == []
  end
end
