# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi.

require 'reinforce/prioritized_experience_replay'

describe Reinforce::PrioritizedExperienceReplay do
  let(:environment) { Object.new }
  let(:q_function_model) { Object.new }
  let(:per) { Reinforce::PrioritizedExperienceReplay.new }
  let(:batch_size) { 4 }
  let(:per_size) { 10 }

  it 'can be instantiated' do
    expect(per).not.to be_nil
  end

  it 'has size 0 at creation time' do
    expect(per.size).to be == 0
  end

  it 'accepts a new experience' do
    per.update(1, 2, 3, 4, 5)
    expect(per.size).to be == 1
  end

  it 'can be sampled' do
    per.update(1, 2, 3, 4, 5)
    expect(per.sample).to be == { state: [1], action: [2], next_state: [3], reward: [4], done: [5] }
  end

  it 'can be sampled with size > 1' do
    input = Array.new(per_size) { [rand, rand, rand, rand, rand] }
    per_size.times { |i| per.update(*input[i]) }

    sample = per.sample(batch_size)
    expect(sample.size == batch_size)

    # rand_index = rand(batch_size)
    # expect(sample[rand_index].size == 5)
    # expect(input.include?(sample[rand_index]))
  end
end
