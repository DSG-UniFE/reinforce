# frozen_string_literal: true

require "reinforce"

describe Reinforce::RunningMeanStd do
  let(:stats) { subject.new(epsilon: 0.0) }

  def population_variance(values)
    mean = values.sum / values.size
    values.sum { |value| (value - mean)**2 } / values.size
  end

  it "starts at mean 0 and variance 1, so normalizing is safe before any update" do
    fresh = subject.new

    expect(fresh.mean).to be_within(1e-12).of(0.0)
    expect(fresh.var).to be_within(1e-12).of(1.0)
    expect(fresh.count).to be > 0
  end

  it "matches the batch mean and population variance of everything seen" do
    values = [3.0, -1.0, 4.0, 1.5, -5.0, 9.0, 2.0]
    stats.update(values)

    expect(stats.mean).to be_within(1e-12).of(values.sum / values.size)
    expect(stats.var).to be_within(1e-12).of(population_variance(values))
    expect(stats.count).to be == values.size
  end

  it "gives the same result whether values arrive one at a time or in batches" do
    values = [0.5, -2.0, 7.0, 3.25, -1.5, 4.0]
    one_by_one = subject.new(epsilon: 0.0)
    values.each { |value| one_by_one.update(value) }
    batched = subject.new(epsilon: 0.0).update(values.first(2)).update(values.drop(2))

    expect(one_by_one.mean).to be_within(1e-12).of(batched.mean)
    expect(one_by_one.var).to be_within(1e-12).of(batched.var)
    expect(one_by_one.std).to be_within(1e-12).of(Math.sqrt(population_variance(values)))
  end

  it "ignores empty batches" do
    stats.update([1.0, 3.0]).update([])

    expect(stats.count).to be == 2
    expect(stats.mean).to be_within(1e-12).of(2.0)
  end
end
