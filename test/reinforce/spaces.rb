# frozen_string_literal: true

require "reinforce/spaces"

describe Reinforce::Spaces::Box do
  let(:space) { subject.new(low: [-1.0, 0.0], high: [1.0, 10.0]) }

  it "expands numeric bounds to every component" do
    box = subject.new(low: -1, high: 1, size: 3)

    expect(box.low).to be == [-1.0, -1.0, -1.0]
    expect(box.high).to be == [1.0, 1.0, 1.0]
    expect(box.size).to be == 3
  end

  it "takes its size from an array bound" do
    expect(subject.new(low: 0.0, high: [1.0, 2.0]).size).to be == 2
  end

  it "rejects inconsistent bounds" do
    expect { subject.new(low: 0.0, high: 1.0) }.to raise_exception(ArgumentError)
    expect { subject.new(low: [0.0], high: [1.0, 2.0]) }.to raise_exception(ArgumentError)
    expect { subject.new(low: [2.0], high: [1.0]) }.to raise_exception(ArgumentError)
  end

  it "samples elements of the space" do
    srand(1)
    samples = Array.new(100) { space.sample }

    expect(samples.all? { |sample| space.contains?(sample) }).to be == true
  end

  it "checks membership" do
    expect(space.contains?([0.0, 5.0])).to be == true
    expect(space.contains?([0.0, 11.0])).to be == false
    expect(space.contains?([0.0])).to be == false
  end

  it "scales from [-1, 1] onto the space and back" do
    expect(space.scale([-1.0, 1.0])).to be == [-1.0, 10.0]
    expect(space.scale([0.0, 0.0])).to be == [0.0, 5.0]
    expect(space.unscale([0.5, 2.5])).to be == [0.5, -0.5]
    expect(subject.new(low: 1.0, high: 1.0, size: 1).unscale([1.0])).to be == [0.0]
  end

  it "compares by value" do
    expect(space).to be == subject.new(low: [-1, 0], high: [1, 10])
    expect(space.hash).to be == subject.new(low: [-1, 0], high: [1, 10]).hash
    expect(space.inspect).to be == "Box(low: [-1.0, 0.0], high: [1.0, 10.0])"
  end
end
