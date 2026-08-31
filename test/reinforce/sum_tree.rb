# frozen_string_literal: true

require "reinforce/sum_tree"

describe Reinforce::SumTree do
  let(:tree) { Reinforce::SumTree.new(4) }

  it "starts with a total of zero" do
    expect(tree.total).to be == 0.0
  end

  it "tracks the sum of all leaf priorities as the total" do
    tree.update(0, 1.0)
    tree.update(1, 2.0)
    tree.update(2, 3.0)

    expect(tree.total).to be == 6.0
  end

  it "lets a later update to the same leaf replace, not add to, its priority" do
    tree.update(0, 5.0)
    tree.update(0, 1.0)

    expect(tree.total).to be == 1.0
  end

  it "finds the leaf whose cumulative range contains a given value" do
    tree.update(0, 1.0)
    tree.update(1, 2.0)
    tree.update(2, 3.0)
    tree.update(3, 4.0)

    # cumulative ranges: [0, 1) -> 0, [1, 3) -> 1, [3, 6) -> 2, [6, 10) -> 3
    expect(tree.get(0.5)).to be == [0, 1.0]
    expect(tree.get(1.5)).to be == [1, 2.0]
    expect(tree.get(4.0)).to be == [2, 3.0]
    expect(tree.get(9.9)).to be == [3, 4.0]
  end

  it "never samples a leaf that was never assigned a priority" do
    tree.update(0, 1.0)
    tree.update(1, 1.0)
    # leaves 2 and 3 are left at their initial priority of 0.0

    100.times do
      data_index, = tree.get(rand * tree.total)
      expect([0, 1].include?(data_index)).to be == true
    end
  end
end
