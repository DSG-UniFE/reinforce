# frozen_string_literal: true

require "reinforce/offline_dataset"
require "reinforce/algorithms/behavior_cloning"
require_relative "../../support/agent_contract"

describe Reinforce::Algorithms::BehaviorCloning do
  include_context AgentContract,
    factory: -> {
      dataset = Reinforce::OfflineDataset.new
      dataset.add(state: [0.0], action: :left, reward: 0.0, next_state: [0.0], done: true)
      dataset.add(state: [0.0], action: :right, reward: 1.0, next_state: [0.0], done: true)
      Reinforce::Algorithms::BehaviorCloning.new(dataset, action_space: %i[left right])
    },
    state: [0.0],
    actions: %i[left right]

  it "learns an ANN policy from offline data and predicts dominant actions" do
    srand(1234)
    Torch.manual_seed(1234)
    dataset = Reinforce::OfflineDataset.new
    60.times { dataset.add(state: [0.0], action: :left, reward: 0, next_state: [0.0], done: false) }
    60.times { dataset.add(state: [1.0], action: :right, reward: 0, next_state: [1.0], done: false) }

    policy = Reinforce::Algorithms::BehaviorCloning.new(dataset, learning_rate: 0.01)
    policy.train(epochs: 120, batch_size: 32)

    expect(policy.logs[:loss].size).to be == 120
    expect(policy.predict([0.0])).to be == :left
    expect(policy.predict([1.0])).to be == :right
  end
end
