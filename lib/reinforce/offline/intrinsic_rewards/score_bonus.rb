# frozen_string_literal: true

require "torch"

module Reinforce
  module Offline
    module IntrinsicRewards
      # Computes intrinsic reward from denoising error.
      class ScoreBonus
        def initialize(noise_std: 0.1)
          @noise_std = noise_std
        end

        def call(score_model, state_action_tensor)
          noisy = state_action_tensor + @noise_std * Torch.randn_like(state_action_tensor)
          reconstructed = Torch.no_grad { score_model.call(noisy) }
          (reconstructed - state_action_tensor).pow(2).mean(dim: 1).reshape(-1)
        end
      end
    end
  end
end
