# frozen_string_literal: true

require 'torch'

module Reinforce
  module Models
    # Learns to denoise state-action vectors; reconstruction error is used
    # as an intrinsic novelty signal in ExDM.
    class DiffusionScoreModel < Torch::NN::Module
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
