# frozen_string_literal: true

require_relative "../../../support/torch_test_helpers"
require "reinforce/models/diffusion_score_model"
require "reinforce/offline/intrinsic_rewards/score_bonus"

describe Reinforce::Offline::IntrinsicRewards::ScoreBonus do
  it "returns one finite novelty score for each state-action vector" do
    TorchTestHelpers.with_torch_seed do
      model = Reinforce::Models::DiffusionScoreModel.new(3, hidden_size: 8)
      state_actions = Torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype: :float32)

      scores = Reinforce::Offline::IntrinsicRewards::ScoreBonus.new(noise_std: 0.05).call(model, state_actions)

      expect(scores.size.to_a).to be == [2]
      expect(TorchTestHelpers.finite_tensor?(scores)).to be == true
      expect(scores.to_a.all? { |score| score >= 0.0 }).to be == true
    end
  end
end
