# frozen_string_literal: true

require 'reinforce/offline_dataset'
require 'reinforce/offline/evaluation'

describe Reinforce::Offline::Evaluation do
  it 'computes weighted importance sampling estimate' do
    episodes = [
      [
        {reward: 1.0, behavior_prob: 0.5, target_prob: 1.0},
        {reward: 1.0, behavior_prob: 0.5, target_prob: 1.0}
      ],
      [
        {reward: 0.0, behavior_prob: 0.5, target_prob: 0.0},
        {reward: 0.0, behavior_prob: 0.5, target_prob: 0.0}
      ]
    ]

    wis = Reinforce::Offline::Evaluation.weighted_importance_sampling(episodes, discount_factor: 1.0)
    expect(wis).to be_within(1e-6).of(2.0)
  end

  it 'fits tabular FQE and estimates policy value' do
    dataset = Reinforce::OfflineDataset.new
    10.times do
      dataset.add(state: [:s0], action: :go, reward: 1.0, next_state: [:s1], done: false, timestep: 0)
      dataset.add(state: [:s1], action: :go, reward: 2.0, next_state: [:s1], done: true, timestep: 1)
    end

    policy = proc { |_state| :go }
    evaluator = Reinforce::Offline::Evaluation::TabularFQE.new(dataset, discount_factor: 0.9, iterations: 30)
    evaluator.fit(policy)
    value = evaluator.policy_value(policy, [[:s0]])

    expect(value > 2.5).to be == true
  end
end
