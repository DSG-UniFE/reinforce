# frozen_string_literal: true

require "reinforce/shields"

describe Reinforce::Shields::BoundedSum do
  let(:aggregator) { subject.new }
  let(:values) { Torch.tensor([[0.2, 0.3], [0.7, 0.6]]) }

  it "sums truth values up to 1" do
    expect(aggregator.call(values, dim: 1).to_a.map { |value| value.round(6) }).to be == [0.5, 1.0]
    expect(aggregator.call(values).item).to be_within(1e-6).of(1.0)
    expect(aggregator.call(values, dim: [0], keepdim: true).shape).to be == [1, 2]
  end

  it "ignores masked-out values" do
    mask = Torch.tensor([[true, false], [false, true]])

    expect(aggregator.call(values, dim: 1, mask: mask).to_a.map { |value| value.round(6) }).to be == [0.2, 0.6]
  end

  it "quantifies existentially in LTN formulas" do
    exists = LTN::Quantifier.new(aggregator, :exists)
    x = LTN::Variable.new("x", Torch.tensor([0.1, 0.2, 0.3]))
    predicate = LTN::Predicate.new { |input| input.reshape(-1) }

    expect(exists.call(x, predicate.call(x)).value.item).to be_within(1e-6).of(0.6)
  end
end
