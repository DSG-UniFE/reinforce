# frozen_string_literal: true

require "torch"

module Reinforce
  # Small, shared building blocks for the neural networks used throughout
  # this library.
  module Networks
    module_function

    # Builds a simple feed-forward (multi-layer perceptron) network:
    # Linear -> activation -> Linear -> activation -> ... -> Linear, with
    # `hidden_layers` activated hidden layers of width `hidden_size`
    # sandwiched between an `input_size -> hidden_size` input layer and a
    # `hidden_size -> output_size` output layer (no activation after the
    # final layer, so the result is suitable for raw logits/Q-values as
    # well as for feeding into e.g. a softmax or a loss function).
    #
    # @parameter input_size [Integer] the size of the input (state) vector.
    # @parameter output_size [Integer] the size of the output vector (e.g. the number of actions).
    # @parameter hidden_size [Integer] the width of each hidden layer.
    # @parameter hidden_layers [Integer] the number of hidden layers (0 gives a single Linear layer, i.e. a linear model).
    # @parameter activation [Class] a Torch::NN module class to instantiate after each hidden layer.
    # @returns [Torch::NN::Sequential] the assembled network.
    def mlp(input_size, output_size, hidden_size: 64, hidden_layers: 2, activation: Torch::NN::ReLU)
      raise ArgumentError, "hidden_layers must be >= 0" if hidden_layers.negative?

      layers = []
      previous_size = input_size
      hidden_layers.times do
        layers << Torch::NN::Linear.new(previous_size, hidden_size)
        layers << activation.new
        previous_size = hidden_size
      end
      layers << Torch::NN::Linear.new(previous_size, output_size)

      Torch::NN::Sequential.new(*layers)
    end

    # Polyak (soft) update: blends `online`'s parameters into `target`'s,
    # in place, as target <- (1 - tau) * target + tau * online.
    #
    # This is the standard way DQN-, SAC-, and CQL-style algorithms keep a
    # target network's bootstrapped values from moving as fast as the
    # network actually being trained, which is what stabilizes training.
    # tau: 1.0 fully copies `online` into `target` (a "hard" update, which
    # is exactly what plain DQN's periodic target-network refresh is);
    # smaller values interpolate more slowly, which is the usual choice
    # for SAC/CQL-style continuous soft updates applied every step.
    #
    # `target` and `online` only need to respond to #state_dict and (for
    # `target`) #load_state_dict, so this works equally well with a raw
    # Torch::NN::Module or with a wrapper object (such as QFunctionANN)
    # that delegates those methods to one.
    #
    # @parameter target [#state_dict, #load_state_dict] the network to update, in place.
    # @parameter online [#state_dict] the network whose parameters are blended in.
    # @parameter tau [Float] the interpolation factor, in the [0.0, 1.0] range.
    # @returns [void]
    def soft_update!(target:, online:, tau:)
      tau = tau.to_f
      raise ArgumentError, "tau must be in the [0.0, 1.0] range" if tau < 0.0 || tau > 1.0

      target_state = target.state_dict
      online_state = online.state_dict

      mixed = {}
      target_state.each do |name, value|
        mixed[name] = (value * (1.0 - tau)) + (online_state[name] * tau)
      end

      target.load_state_dict(mixed)
    end
  end
end
