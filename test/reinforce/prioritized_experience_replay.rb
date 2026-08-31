# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi.

require "reinforce/prioritized_experience_replay"

describe Reinforce::PrioritizedExperienceReplay do
  let(:per) { Reinforce::PrioritizedExperienceReplay.new }
  let(:batch_size) { 4 }
  let(:per_size) { 10 }

  it "can be instantiated" do
    expect(per).not.to be_nil
  end

  it "has size 0 at creation time" do
    expect(per.size).to be == 0
  end

  it "accepts a new experience" do
    per.update(1, 2, 3, 4, 5)
    expect(per.size).to be == 1
  end

  it "raises when sampling from an empty buffer" do
    expect { per.sample }.to raise_exception(ArgumentError)
  end

  it "can be sampled" do
    per.update(1, 2, 3, 4, 5)
    sample = per.sample

    expect(sample[:state]).to be == [1]
    expect(sample[:action]).to be == [2]
    expect(sample[:next_state]).to be == [3]
    expect(sample[:reward]).to be == [4]
    expect(sample[:done]).to be == [5]
    expect(sample[:indices]).to be == [0]
    expect(sample[:weights]).to be == [1.0]
  end

  it "can be sampled with size > 1" do
    input = Array.new(per_size) { [rand, rand, rand, rand, rand] }
    per_size.times { |i| per.update(*input[i]) }

    sample = per.sample(batch_size)

    expect(sample[:state].size).to be == batch_size
    expect(sample[:indices].size).to be == batch_size
    expect(sample[:weights].size).to be == batch_size
    sample[:indices].each { |index| expect((0...per_size).cover?(index)).to be == true }
  end

  it "normalizes importance-sampling weights so the largest one is 1.0" do
    per_size.times { |i| per.update(i, i, i, i, false) }
    per.update_priorities([0, 1], [10.0, 0.001])

    weights = per.sample(per_size)[:weights]

    expect(weights.max).to be_within(1e-9).of(1.0)
    weights.each { |weight| expect(weight <= 1.0).to be == true }
  end

  it "handles negative rewards without producing a degenerate sampling distribution" do
    # Regression test: the previous implementation sampled with probability
    # proportional to raw reward, which broke (a negative-weight cumulative
    # distribution) as soon as any stored reward was negative. Priority is
    # now derived from TD-error magnitude, not reward, so it is always
    # positive regardless of the reward's sign.
    per.update(0, 0, 0, -5.0, false)
    per.update(1, 1, 1, -1.0, false)
    per.update(2, 2, 2, 3.0, false)

    sample = per.sample(3)

    expect(sample[:reward].sort).to be == [-5.0, -1.0, 3.0]
    sample[:weights].each { |weight| expect(weight > 0).to be == true }
  end

  it "gives every transition a chance to be sampled before any priority update" do
    # New transitions default to the maximum priority seen so far, so a
    # freshly-added transition isn't starved of training just because no
    # TD-error has been computed for it yet.
    per_size.times { |i| per.update(i, i, i, i, false) }

    seen = Set.new
    50.times { seen.merge(per.sample(per_size)[:indices]) }

    expect(seen.size).to be == per_size
  end

  it "samples transitions with a larger reported TD-error more often" do
    2.times { |i| per.update(i, i, i, i, false) }
    per.update_priorities([0, 1], [100.0, 0.0001])

    counts = Hash.new(0)
    200.times { per.sample[:indices].each { |index| counts[index] += 1 } }

    expect(counts[0] > counts[1]).to be == true
  end

  it "accepts bulk updates" do
    experience = {
      state: [0, 1],
      action: [0, 1],
      next_state: [0, 1],
      reward: [0.0, 1.0],
      done: [false, false]
    }

    per.bulk_update(experience)

    expect(per.size).to be == 2
  end
end
