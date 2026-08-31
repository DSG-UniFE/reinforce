# frozen_string_literal: true

require_relative "../support/torch_test_helpers"
require "reinforce/networks"

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
