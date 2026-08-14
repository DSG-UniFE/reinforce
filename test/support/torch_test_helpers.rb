# frozen_string_literal: true

require "torch"

module TorchTestHelpers
  module_function

  DEFAULT_SEED = 12_345

  def with_torch_seed(seed = test_seed)
    srand(seed)
    Torch.manual_seed(seed)
    yield(seed)
  end

  def test_seed
    Integer(ENV.fetch("REINFORCE_TEST_SEED", DEFAULT_SEED.to_s))
  end

  def parameter_snapshot(model)
    model.parameters.map { |parameter| parameter.detach.to_a }
  end

  def parameters_changed?(model, snapshot)
    parameter_snapshot(model) != snapshot
  end

  def finite_tensor?(tensor)
    tensor.to_a.flatten.all? { |value| value.finite? }
  end
end
