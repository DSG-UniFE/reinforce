# frozen_string_literal: true

require 'torch'

module Reinforce
  module Models
    # Minimal diffusion-style policy backbone for discrete actions.
    # In this first draft, it is an MLP policy used by ExDM updates.
    class DiffusionPolicy < Torch::NN::Module
      attr_reader :action_space

      def initialize(state_size, action_space, hidden_size: 64)
        super()
        @action_space = action_space
        @network = Torch::NN::Sequential.new(
          Torch::NN::Linear.new(state_size, hidden_size),
          Torch::NN::ReLU.new,
          Torch::NN::Linear.new(hidden_size, hidden_size),
          Torch::NN::ReLU.new,
          Torch::NN::Linear.new(hidden_size, @action_space.size)
        )
      end

      def forward(states)
        @network.call(states)
      end

      def probabilities(states)
        Torch.softmax(forward(states), dim: 1)
      end

      def predict(state)
        argument = state.is_a?(Torch::Tensor) ? state : Torch.tensor([state], dtype: :float32)
        logits = Torch.no_grad { forward(argument) }
        @action_space[logits.argmax(1).item]
      end
    end
  end
end
