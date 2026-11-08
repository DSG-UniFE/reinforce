# frozen_string_literal: true

require "torch"
require_relative "device"

module Reinforce
  # Small, shared building blocks for the neural networks used throughout
  # this library.
  module Networks
    module_function

    # Builds just the stack of hidden Linear+activation layers -- no final
    # output projection -- for a network that branches into more than one
    # head after a shared feature extractor (e.g.
    # Reinforce::Models::DuelingQNetwork's value/advantage streams). #mlp
    # is itself built from this plus one final Linear layer.
    #
    # `hidden_size` accepts either an Integer (every hidden layer gets
    # this width; `hidden_layers` says how many) or an Array of Integers
    # (one width per layer, in the given order -- `hidden_layers` is then
    # ignored, since the array's length already says how many layers
    # there are). `hidden_layers: 0` (or `hidden_size: []`) means no
    # hidden layers at all.
    #
    # @parameter input_size [Integer] the size of the input vector.
    # @parameter hidden_size [Integer, Array(Integer)] the width of each hidden layer, or the exact per-layer widths.
    # @parameter hidden_layers [Integer] the number of hidden layers (ignored if hidden_size is an Array).
    # @parameter activation [Class] a Torch::NN module class to instantiate after each hidden layer.
    # @returns [Torch::NN::Sequential] the assembled stack of hidden layers.
    def trunk(input_size, hidden_size: 64, hidden_layers: 2, activation: Torch::NN::ReLU)
      layers, _output_size = build_hidden_layers(input_size, hidden_size, hidden_layers, activation)
      Torch::NN::Sequential.new(*layers)
    end

    # Builds a simple feed-forward (multi-layer perceptron) network:
    # Linear -> activation -> Linear -> activation -> ... -> Linear (see
    # #trunk for the hidden portion), ending in an `-> output_size` layer
    # with no activation after it, so the result is suitable for raw
    # logits/Q-values as well as for feeding into e.g. a softmax or a loss
    # function.
    #
    # @parameter input_size [Integer] the size of the input (state) vector.
    # @parameter output_size [Integer] the size of the output vector (e.g. the number of actions).
    # @parameter hidden_size [Integer, Array(Integer)] see #trunk.
    # @parameter hidden_layers [Integer] see #trunk.
    # @parameter activation [Class] a Torch::NN module class to instantiate after each hidden layer.
    # @returns [Torch::NN::Sequential] the assembled network.
    def mlp(input_size, output_size, hidden_size: 64, hidden_layers: 2, activation: Torch::NN::ReLU)
      layers, trunk_output_size = build_hidden_layers(input_size, hidden_size, hidden_layers, activation)
      layers << Torch::NN::Linear.new(trunk_output_size, output_size)
      Torch::NN::Sequential.new(*layers)
    end

    # Shared implementation for #trunk and #mlp, also useful directly for
    # a multi-head network that needs both the hidden layers themselves
    # and their output width in order to size its own heads (see
    # Reinforce::Models::DuelingQNetwork) -- #trunk alone only returns the
    # assembled Sequential, losing the output-width information.
    #
    # @returns [Array(Array(Torch::NN::Module), Integer)] the layer
    #   objects (Linear/activation, interleaved) and the width a caller
    #   should build the next layer from.
    def build_hidden_layers(input_size, hidden_size, hidden_layers, activation)
      widths = if hidden_size.is_a?(Array)
        hidden_size
      else
        raise ArgumentError, "hidden_layers must be >= 0" if hidden_layers.negative?

        Array.new(hidden_layers, hidden_size)
      end

      layers = []
      previous_size = input_size
      widths.each do |width|
        layers << Torch::NN::Linear.new(previous_size, width)
        layers << activation.new
        previous_size = width
      end

      [layers, previous_size]
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

    # A small, constructible-and-diffable description of an MLP's shape.
    # Algorithms that build a default network internally (unless the
    # caller supplies a full `architecture:`/`*_model:` override -- see
    # e.g. QFunctionANN, DQN, ConservativeQLearning) accept one of these
    # as `network_config:`, so reshaping that default is "construct one
    # value object," not "write your own Torch::NN::Module subclass."
    # The default (64, 2, ReLU, "cpu") matches what every algorithm in
    # this library did before this existed.
    #
    # @parameter hidden_size [Integer, Array(Integer)] see Networks#trunk/#mlp.
    # @parameter hidden_layers [Integer] see Networks#trunk/#mlp.
    # @parameter activation [Class] a Torch::NN module class to instantiate after each hidden layer.
    # @parameter device [Symbol, String] see Reinforce::Device.resolve --
    #   #build moves the network it returns onto this device. Note this
    #   only makes an algorithm's *model* device-aware; whether that
    #   algorithm's training-loop tensors are also built on the same
    #   device (required for it to actually run there) currently varies
    #   by algorithm -- see QFunctionANN/DQN for the one path that's
    #   fully wired end to end today. Others will raise Torch's ordinary
    #   "expected all tensors to be on the same device" error the moment
    #   a mismatched tensor meets this network, which is the correct,
    #   self-diagnosing failure mode for a not-yet-migrated caller, not a
    #   bug to work around.
    Config = Data.define(:hidden_size, :hidden_layers, :activation, :device) do
      def initialize(hidden_size: 64, hidden_layers: 2, activation: Torch::NN::ReLU, device: "cpu")
        super
      end

      # Build a Torch::NN::Sequential MLP of this shape, moved onto
      # `device` -- see Networks#mlp, which this delegates to for the
      # architecture itself.
      def build(input_size, output_size)
        # Torch::NN::Module#to takes device as a positional argument, not
        # a device: keyword (unlike Torch.tensor(..., device:)) -- easy
        # to get wrong, since it fails with a confusing "does not wrap a
        # C++ object" error rather than an ArgumentError.
        Networks.mlp(input_size, output_size, hidden_size:, hidden_layers:, activation:).to(Reinforce::Device.resolve(device))
      end
    end

    # A residual ("skip connection") wrapper (He et al., 2016): computes
    # `input + inner.call(input)` instead of just `inner.call(input)`,
    # adding a learned projection first if `inner`'s output width doesn't
    # match `input`'s width (so the addition is shape-valid). This is the
    # core ResNet primitive -- it helps gradients flow through deep
    # stacks of layers, which is useful even for a plain deep MLP
    # policy/value network, not just image models.
    #
    # `inner` can be anything that responds to #call -- typically a
    # Reinforce::Networks.trunk, but a hand-written Torch::NN::Module
    # works too. Compose it with whatever you're already building rather
    # than treating it as a whole network on its own.
    class ResidualBlock < Torch::NN::Module
      def initialize(inner, input_size:, output_size: input_size)
        super()
        @inner = inner
        @projection = (input_size == output_size) ? nil : Torch::NN::Linear.new(input_size, output_size)
      end

      def forward(input)
        skip = @projection ? @projection.call(input) : input
        skip + @inner.call(input)
      end
    end

    # An encoder/decoder pair around a bottleneck representation.
    # #forward reconstructs (`decoder.call(encoder.call(input))`); #encode
    # exposes just the bottleneck on its own, which a single opaque
    # Sequential can't offer -- there's no point mid-pipeline to read
    # from. This is the shape every autoencoder needs; see
    # Reinforce::Models::DenoisingAutoencoder for the one already built
    # from it.
    #
    # `encoder`/`decoder` can be anything that responds to #call --
    # typically the result of Networks#trunk/#mlp/Config#build, but a
    # hand-written Torch::NN::Module works on either side too, e.g. if
    # you need a convolutional encoder that Networks#trunk can't express.
    class EncoderDecoder < Torch::NN::Module
      attr_reader :encoder, :decoder

      def initialize(encoder, decoder)
        super()
        @encoder = encoder
        @decoder = decoder
      end

      def encode(input)
        @encoder.call(input)
      end

      def forward(input)
        @decoder.call(encode(input))
      end
    end
  end
end
