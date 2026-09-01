# frozen_string_literal: true

require_relative "../../support/torch_test_helpers"
require "reinforce/models/denoising_autoencoder"

describe Reinforce::Models::DenoisingAutoencoder do
  it "maps noisy state-action vectors to finite reconstructions" do
    TorchTestHelpers.with_torch_seed do
      model = Reinforce::Models::DenoisingAutoencoder.new(3, hidden_size: 8)
      inputs = Torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype: :float32)

      outputs = model.call(inputs)

      expect(outputs.size.to_a).to be == [2, 3]
      expect(TorchTestHelpers.finite_tensor?(outputs)).to be == true
    end
  end
end
