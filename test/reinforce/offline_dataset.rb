# frozen_string_literal: true

require "reinforce/offline_dataset"

describe Reinforce::OfflineDataset do
  it "stores valid transitions and reports dataset size" do
    dataset = Reinforce::OfflineDataset.new
    dataset.add(state: [0], action: :a, reward: 1, next_state: [1], done: false, timestep: 0)
    dataset.add(state: [1], action: :b, reward: 2, next_state: [2], done: true, timestep: 1)

    expect(dataset.size).to be == 2
    expect(dataset.empty?).to be == false
  end

  it "raises when required fields are missing" do
    dataset = Reinforce::OfflineDataset.new
    expect do
      dataset.add(state: [0], action: :a, reward: 1)
    end.to raise_exception(ArgumentError)
  end

  it "samples mini-batches and keeps transition structure" do
    dataset = Reinforce::OfflineDataset.new
    5.times do |i|
      dataset.add(state: [i], action: :a, reward: i, next_state: [i + 1], done: false)
    end

    batch = dataset.sample(3, random: Random.new(1234))
    expect(batch.size).to be == 3
    expect(batch.all? { |transition| transition.key?(:state) && transition.key?(:action) }).to be == true
  end

  it "extracts initial states from timestep tags" do
    dataset = Reinforce::OfflineDataset.new
    dataset.add(state: [0], action: :a, reward: 0, next_state: [1], done: false, timestep: 0)
    dataset.add(state: [1], action: :a, reward: 0, next_state: [2], done: false, timestep: 1)
    dataset.add(state: [9], action: :a, reward: 0, next_state: [9], done: false, timestep: 0)

    expect(dataset.initial_states).to be == [[0], [9]]
  end
end
