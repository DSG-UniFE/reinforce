# frozen_string_literal: true

require_relative "../../support/torch_test_helpers"
require "reinforce/models/denoising_autoencoder"
require "reinforce/networks"

describe Reinforce::Models::DenoisingAutoencoder do
  it "maps noisy state-action vectors to finite reconstructions" do
    TorchTestHelpers.with_torch_seed do
      model = Reinforce::Models::DenoisingAutoencoder.new(3, network_config: Reinforce::Networks::Config.new(hidden_size: 8))
      inputs = Torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype: :float32)

      outputs = model.call(inputs)

      expect(outputs.size.to_a).to be == [2, 3]
      expect(TorchTestHelpers.finite_tensor?(outputs)).to be == true
    end
  end

  it "#encode exposes the bottleneck representation on its own" do
    # Regression test: the previous implementation was one opaque
    # Sequential with no way to read the intermediate (encoded)
    # representation -- only the full reconstruction.
    TorchTestHelpers.with_torch_seed do
      model = Reinforce::Models::DenoisingAutoencoder.new(3, network_config: Reinforce::Networks::Config.new(hidden_size: 8))
      inputs = Torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype: :float32)

      encoded = model.encode(inputs)

      # The bottleneck is the trunk's own output width (8), not input_size
      # (3) -- #forward projects back down to input_size afterwards.
      expect(encoded.size.to_a).to be == [2, 8]
      expect(TorchTestHelpers.finite_tensor?(encoded)).to be == true
    end
  end
end
