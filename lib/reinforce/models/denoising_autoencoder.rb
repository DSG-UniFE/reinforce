# frozen_string_literal: true

require "torch"
require_relative "../networks"

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
      def initialize(input_size, network_config: Reinforce::Networks::Config.new)
        super()
        # Built directly from Networks.build_hidden_layers, rather than
        # network_config.build(input_size, input_size), specifically so
        # the encoder/decoder split -- and #encode -- exist at all. This
        # reproduces the exact same layers as network_config.build would
        # (trunk + one final Linear back to input_size), just wrapped as
        # a Reinforce::Networks::EncoderDecoder instead of one opaque
        # Sequential, so the bottleneck (the trunk's own output) is
        # readable on its own via #encode, not just as a reconstruction.
        layers, trunk_output_size = Reinforce::Networks.build_hidden_layers(
          input_size, network_config.hidden_size, network_config.hidden_layers, network_config.activation
        )
        encoder = Torch::NN::Sequential.new(*layers)
        decoder = Torch::NN::Linear.new(trunk_output_size, input_size)
        @encoder_decoder = Reinforce::Networks::EncoderDecoder.new(encoder, decoder)
        to(Reinforce::Device.resolve(network_config.device))
      end

      def encode(inputs)
        @encoder_decoder.encode(inputs)
      end

      def forward(noisy_inputs)
        @encoder_decoder.call(noisy_inputs)
      end
    end
  end
end
