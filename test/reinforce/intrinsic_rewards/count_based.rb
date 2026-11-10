# frozen_string_literal: true

require "reinforce"

describe Reinforce::IntrinsicRewards::CountBased do
  let(:bonus) { subject.new(coefficient: 0.5) }

  it "pays coefficient / sqrt(visit count), treating unseen states as seen once" do
    4.times { bonus.observe([1, 1]) }
    bonus.observe([2, 2])

    values = bonus.call([[1, 1], [2, 2], [3, 3]])

    expect(values[0]).to be_within(1e-12).of(0.5 / 2)
    expect(values[1]).to be_within(1e-12).of(0.5)
    expect(values[2]).to be_within(1e-12).of(0.5)
    expect([bonus.count([1, 1]), bonus.count([3, 3])]).to be == [4, 0]
  end

  it "counts states by their Array form, so equal Tensors and Arrays share a count" do
    bonus.observe(Torch.tensor([1, 2]))
    bonus.observe([1, 2])

    expect(bonus.count([1, 2])).to be == 2
  end

  it "counts by a custom key, e.g. to discretize continuous states" do
    rounded = subject.new(key: ->(state) { state.map(&:round) })
    rounded.observe([0.9, 2.1])
    rounded.observe([1.2, 1.8])

    expect(rounded.count([1, 2])).to be == 2
  end

  it "has nothing to train" do
    expect(bonus.train([[1, 1]])).to be_equal(bonus)
  end
end
