# frozen_string_literal: true

require 'reinforce/offline_dataset'
require 'reinforce/offline/evaluation'
require 'reinforce/algorithms/behavior_cloning'
require 'reinforce/algorithms/conservative_q_learning'

describe 'offline RL pipeline integration' do
  it 'trains BC and CQL from one dataset and evaluates with FQE' do
    srand(1234)
    Torch.manual_seed(1234)
    dataset = Reinforce::OfflineDataset.new

    80.times do
      dataset.add(state: [0.0], action: :good, reward: 1.0, next_state: [1.0], done: true, timestep: 0)
    end
    20.times do
      dataset.add(state: [0.0], action: :bad, reward: 0.0, next_state: [1.0], done: true, timestep: 0)
    end

    bc = Reinforce::Algorithms::BehaviorCloning.new(dataset, action_space: %i[good bad], learning_rate: 0.01)
    bc.train(epochs: 100, batch_size: 32)

    cql = Reinforce::Algorithms::ConservativeQLearning.new(
      dataset,
      action_space: %i[good bad],
      learning_rate: 0.01,
      discount_factor: 0.9,
      alpha: 0.1
    )
    cql.train(epochs: 80, batch_size: 32, tau: 0.2)

    fqe = Reinforce::Offline::Evaluation::TabularFQE.new(dataset, discount_factor: 0.9, iterations: 30)
    fqe.fit(proc { |state| bc.predict(state) })
    bc_value = fqe.policy_value(proc { |state| bc.predict(state) }, [[0.0]])

    expect(bc.predict([0.0])).to be == :good
    expect(cql.predict([0.0])).to be == :good
    expect(bc_value > 0.7).to be == true
  end
end
