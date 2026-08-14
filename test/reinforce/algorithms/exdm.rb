# frozen_string_literal: true

require_relative "../../support/torch_test_helpers"
require "reinforce/offline_dataset"
require "reinforce/algorithms/exdm"

describe Reinforce::Algorithms::ExDM do
  it "trains on offline data and shifts policy toward higher reward action" do
    TorchTestHelpers.with_torch_seed do
      dataset = Reinforce::OfflineDataset.new
      120.times do
        dataset.add(state: [0.0], action: :good, reward: 1.0, next_state: [0.0], done: true)
        dataset.add(state: [0.0], action: :bad, reward: 0.0, next_state: [0.0], done: true)
      end

      exdm = Reinforce::Algorithms::ExDM.new(
        dataset,
        action_space: %i[good bad],
        learning_rate: 0.01,
        discount_factor: 0.95,
        entropy_coefficient: 0.05,
        intrinsic_coefficient: 0.05,
        hidden_size: 32
      )

      before_action = exdm.predict([0.0])
      exdm.train(epochs: 80, batch_size: 64, tau: 0.2)
      after_action = exdm.predict([0.0])

      expect(exdm.logs[:loss].size).to be == 80
      expect(exdm.logs[:policy_loss].size).to be == 80
      expect(exdm.logs[:q_loss].size).to be == 80
      expect(exdm.logs[:score_loss].size).to be == 80
      expect(before_action.nil?).to be == false
      expect(after_action).to be == :good
    end
  end
end
