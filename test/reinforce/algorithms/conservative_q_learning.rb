# frozen_string_literal: true

require "reinforce/offline_dataset"
require "reinforce/algorithms/conservative_q_learning"

describe Reinforce::Algorithms::ConservativeQLearning do
  it "learns ANN Q-values that prefer the higher reward action" do
    srand(1234)
    Torch.manual_seed(1234)
    dataset = Reinforce::OfflineDataset.new
    40.times do
      dataset.add(state: [0.0], action: :good, reward: 2.0, next_state: [0.0], done: true)
      dataset.add(state: [0.0], action: :bad, reward: 0.5, next_state: [0.0], done: true)
    end

    learner = Reinforce::Algorithms::ConservativeQLearning.new(
      dataset,
      action_space: %i[good bad],
      learning_rate: 0.01,
      discount_factor: 0.9,
      alpha: 0.1
    )
    learner.train(epochs: 80, batch_size: 32, tau: 0.2)

    expect(learner.logs[:loss].size).to be == 80
    expect(learner.value([0.0], :good) > learner.value([0.0], :bad)).to be == true
    expect(learner.predict([0.0])).to be == :good
  end
end
