# frozen_string_literal: true

require_relative "../support/torch_test_helpers"
require "reinforce/networks"
require "reinforce/device"

describe Reinforce::Networks do
  describe ".mlp" do
    it "builds a network with the requested number of hidden layers and widths" do
      TorchTestHelpers.with_torch_seed do
        network = Reinforce::Networks.mlp(3, 2, hidden_size: 8, hidden_layers: 2)
        # Linear(3, 8) -> ReLU -> Linear(8, 8) -> ReLU -> Linear(8, 2):
        # 3 Linear layers, each with a weight and a bias, so 6 parameter
        # tensors in total.
        expect(network.parameters.size).to be == 6

        output = network.call(Torch.tensor([[0.0, 1.0, 2.0]], dtype: :float32))
        expect(output.size.to_a).to be == [1, 2]
        expect(TorchTestHelpers.finite_tensor?(output)).to be == true
      end
    end

    it "builds a single Linear layer when hidden_layers is 0" do
      network = Reinforce::Networks.mlp(3, 2, hidden_layers: 0)

      # Just Linear(3, 2): one weight tensor and one bias tensor.
      expect(network.parameters.size).to be == 2

      output = network.call(Torch.tensor([[0.0, 1.0, 2.0]], dtype: :float32))
      expect(output.size.to_a).to be == [1, 2]
    end

    it "rejects a negative number of hidden layers" do
      expect do
        Reinforce::Networks.mlp(3, 2, hidden_layers: -1)
      end.to raise_exception(ArgumentError)
    end

    it "accepts an Array hidden_size for per-layer widths, ignoring hidden_layers" do
      network = Reinforce::Networks.mlp(3, 2, hidden_size: [8, 4], hidden_layers: 99)

      # Linear(3, 8) -> ReLU -> Linear(8, 4) -> ReLU -> Linear(4, 2): three
      # Linear layers regardless of the (ignored) hidden_layers: 99.
      expect(network.parameters.size).to be == 6

      output = network.call(Torch.tensor([[0.0, 1.0, 2.0]], dtype: :float32))
      expect(output.size.to_a).to be == [1, 2]
    end
  end

  describe ".trunk" do
    it "builds only the hidden layers, with no final output projection" do
      trunk = Reinforce::Networks.trunk(3, hidden_size: 8, hidden_layers: 2)

      # Linear(3, 8) -> ReLU -> Linear(8, 8) -> ReLU: two Linear layers,
      # no third one projecting down to some output size.
      expect(trunk.parameters.size).to be == 4

      output = trunk.call(Torch.tensor([[0.0, 1.0, 2.0]], dtype: :float32))
      # The trunk's output width is the last hidden layer's width (8),
      # not some separate output_size -- #trunk doesn't take one.
      expect(output.size.to_a).to be == [1, 8]
    end

    it "is what #mlp is built from -- an empty trunk gives a single Linear layer" do
      trunk = Reinforce::Networks.trunk(3, hidden_layers: 0)

      expect(trunk.parameters.size).to be == 0

      output = trunk.call(Torch.tensor([[0.0, 1.0, 2.0]], dtype: :float32))
      expect(output.size.to_a).to be == [1, 3]
    end
  end

  describe "Config" do
    it "defaults to hidden_size: 64, hidden_layers: 2, activation: Torch::NN::ReLU, device: cpu" do
      config = Reinforce::Networks::Config.new

      expect(config.hidden_size).to be == 64
      expect(config.hidden_layers).to be == 2
      expect(config.activation).to be == Torch::NN::ReLU
      expect(config.device).to be == "cpu"
    end

    it "#build moves the network onto the requested device" do
      # This machine's actual device story (CPU-only, or with an
      # accelerator) doesn't matter here -- explicitly requesting "cpu"
      # must always work and always land there, everywhere.
      config = Reinforce::Networks::Config.new(hidden_size: 4, device: "cpu")

      network = config.build(3, 2)

      network.parameters.each do |parameter|
        expect(parameter.device.to_s).to be == "cpu"
      end
    end

    it "#build resolves device: :auto the same way Reinforce::Device does" do
      # Environment-adaptive rather than hardcoding an expected device --
      # this must land wherever this machine's own best accelerator is
      # (which could be "cpu", "cuda", or "mps" depending on what's
      # actually running the suite), not one specific value.
      config = Reinforce::Networks::Config.new(hidden_size: 4, device: :auto)

      network = config.build(3, 2)

      expected_device = Reinforce::Device.resolve(:auto)
      network.parameters.each do |parameter|
        # Torch reports a concrete device as "type:index" (e.g. "mps:0")
        # even when only the type ("mps") was requested, so compare on
        # the type prefix rather than exact equality.
        expect(parameter.device.to_s.start_with?(expected_device)).to be == true
      end
    end

    it "#build delegates to Networks.mlp with its own fields" do
      config = Reinforce::Networks::Config.new(hidden_size: 8, hidden_layers: 1)

      network = config.build(3, 2)

      # Linear(3, 8) -> ReLU -> Linear(8, 2): two Linear layers.
      expect(network.parameters.size).to be == 4

      output = network.call(Torch.tensor([[0.0, 1.0, 2.0]], dtype: :float32))
      expect(output.size.to_a).to be == [1, 2]
    end

    it "accepts an Array hidden_size, same as Networks.mlp/#trunk" do
      config = Reinforce::Networks::Config.new(hidden_size: [8, 4])

      network = config.build(3, 2)

      expect(network.parameters.size).to be == 6
    end
  end

  describe Reinforce::Networks::ResidualBlock do
    it "computes input + inner.call(input) when widths already match" do
      inner = Torch::NN::Linear.new(2, 2)
      Torch::NN::Init.constant!(inner.weight, 0.0)
      Torch::NN::Init.constant!(inner.bias, 1.0)
      block = Reinforce::Networks::ResidualBlock.new(inner, input_size: 2)

      output = block.call(Torch.tensor([[3.0, 4.0]], dtype: :float32))

      # No projection needed (input_size == output_size), so the skip
      # path is the raw input: [3.0, 4.0] + inner([3.0, 4.0]) == [3.0,
      # 4.0] + [1.0, 1.0] (weight is zero, bias is 1.0) == [4.0, 5.0].
      expect(output.to_a.first).to be == [4.0, 5.0]
      # A skip connection with matching widths needs no extra learned
      # projection.
      expect(block.parameters.size).to be == inner.parameters.size
    end

    it "adds a learned projection to the skip path when widths differ" do
      inner = Torch::NN::Linear.new(2, 5)
      block = Reinforce::Networks::ResidualBlock.new(inner, input_size: 2, output_size: 5)

      output = block.call(Torch.tensor([[0.0, 1.0]], dtype: :float32))

      expect(output.size.to_a).to be == [1, 5]
      # inner's own weight+bias, plus a projection Linear(2, 5)'s
      # weight+bias.
      expect(block.parameters.size).to be == inner.parameters.size + 2
    end
  end

  describe Reinforce::Networks::EncoderDecoder do
    it "#forward reconstructs via decoder(encoder(input))" do
      encoder = Torch::NN::Linear.new(4, 2)
      decoder = Torch::NN::Linear.new(2, 4)
      model = Reinforce::Networks::EncoderDecoder.new(encoder, decoder)

      output = model.call(Torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype: :float32))

      expect(output.size.to_a).to be == [1, 4]
    end

    it "#encode exposes just the bottleneck, not the full reconstruction" do
      encoder = Torch::NN::Linear.new(4, 2)
      decoder = Torch::NN::Linear.new(2, 4)
      model = Reinforce::Networks::EncoderDecoder.new(encoder, decoder)
      input = Torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype: :float32)

      bottleneck = model.encode(input)

      expect(bottleneck.size.to_a).to be == [1, 2]
      # #encode must be the literal first half of #forward, not a
      # separate computation -- decoding it should reproduce #forward's
      # output exactly.
      expect(model.call(input).to_a).to be == decoder.call(bottleneck).to_a
    end
  end

  describe ".soft_update!" do
    def constant_linear(input_size, output_size, value)
      layer = Torch::NN::Linear.new(input_size, output_size)
      Torch::NN::Init.constant!(layer.weight, value)
      Torch::NN::Init.constant!(layer.bias, value)
      layer
    end

    it "blends target <- (1 - tau) * target + tau * online" do
      target = constant_linear(2, 2, 1.0)
      online = constant_linear(2, 2, 3.0)

      Reinforce::Networks.soft_update!(target: target, online: online, tau: 0.25)

      # (1 - 0.25) * 1.0 + 0.25 * 3.0 == 1.5
      target.parameters.each do |parameter|
        parameter.flatten.to_a.each do |value|
          expect(value).to be_within(1e-5).of(1.5)
        end
      end
    end

    it "fully copies online into target when tau is 1.0 (a hard update)" do
      target = constant_linear(2, 2, 1.0)
      online = constant_linear(2, 2, 3.0)

      Reinforce::Networks.soft_update!(target: target, online: online, tau: 1.0)

      target.parameters.each do |parameter|
        parameter.flatten.to_a.each do |value|
          expect(value).to be_within(1e-5).of(3.0)
        end
      end
    end

    it "leaves target unchanged when tau is 0.0" do
      target = constant_linear(2, 2, 1.0)
      online = constant_linear(2, 2, 3.0)

      Reinforce::Networks.soft_update!(target: target, online: online, tau: 0.0)

      target.parameters.each do |parameter|
        parameter.flatten.to_a.each do |value|
          expect(value).to be_within(1e-5).of(1.0)
        end
      end
    end

    it "rejects a tau outside the [0.0, 1.0] range" do
      target = constant_linear(2, 2, 1.0)
      online = constant_linear(2, 2, 3.0)

      expect do
        Reinforce::Networks.soft_update!(target: target, online: online, tau: 1.5)
      end.to raise_exception(ArgumentError)
    end
  end
end
