# frozen_string_literal: true

require "torch"

module Reinforce
  module Models
    # A single-step denoising autoencoder (Vincent et al., 2008): learns to
    # reconstruct a state-action vector from one noised version of it;
    # reconstruction error is used as an intrinsic novelty signal in ExDM
    # (see Reinforce::Offline::IntrinsicRewards::ScoreBonus). Named
    # DenoisingAutoencoder, not DiffusionScoreModel -- there's no noise
    # schedule or timestep conditioning here, just one fixed noise level
    # and one reconstruction pass, so "diffusion" overstated what this is.
    # Structurally this is closer to Random Network Distillation's
    # predictor network than to an actual diffusion model.
    class DenoisingAutoencoder < Torch::NN::Module
      def initialize(input_size, hidden_size: 64)
        super()
        @network = Torch::NN::Sequential.new(
          Torch::NN::Linear.new(input_size, hidden_size),
          Torch::NN::ReLU.new,
          Torch::NN::Linear.new(hidden_size, hidden_size),
          Torch::NN::ReLU.new,
          Torch::NN::Linear.new(hidden_size, input_size)
        )
      end

      def forward(noisy_inputs)
        @network.call(noisy_inputs)
      end
    end
  end
end
