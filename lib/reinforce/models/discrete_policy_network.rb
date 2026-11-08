# frozen_string_literal: true

require "torch"
require_relative "../networks"

module Reinforce
  module Models
    # A plain softmax MLP policy for discrete actions, used by ExDM's SAC
    # core. Named DiscretePolicyNetwork, not DiffusionPolicy -- despite the
    # name it previously had, there was never anything diffusion-related
    # here (no noise, no timestep conditioning, no iterative sampling).
    class DiscretePolicyNetwork < Torch::NN::Module
      attr_reader :action_space

      def initialize(state_size, action_space, network_config: Reinforce::Networks::Config.new)
        super()
        @action_space = action_space
        # network_config.build already moves @network onto the resolved
        # device; resolve it the same way here too, so #predict's own
        # tensor construction stays consistent with it.
        @device = Reinforce::Device.resolve(network_config.device)
        @network = network_config.build(state_size, @action_space.size)
      end

      def forward(states)
        @network.call(states)
      end

      def probabilities(states)
        Torch.softmax(forward(states), dim: 1)
      end

      # @parameter deterministic [Boolean] greedy if true, sampled if false.
      # @returns [Object] an element of the action space.
      def predict(state, deterministic: true)
        argument = state.is_a?(Torch::Tensor) ? state.to(@device) : Torch.tensor([state], dtype: :float32, device: @device)
        logits = Torch.no_grad { forward(argument) }
        @action_space[CategoricalDistribution.new(logits: logits).choose(deterministic: deterministic)]
      end
    end
  end
end
