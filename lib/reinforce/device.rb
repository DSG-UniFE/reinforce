# frozen_string_literal: true

require "torch"

module Reinforce
  # Resolves a device specification into a string Torch::Tensor#to and
  # Torch::NN::Module#to accept directly, so callers don't have to
  # hand-detect what accelerator (if any) is actually available.
  module Device
    module_function

    # @parameter spec [Symbol, String] `:auto` (the default) picks the
    #   best available device: CUDA if present, then Apple Silicon's MPS,
    #   then plain CPU. Anything else (`:cpu`, `:cuda`, `:mps`, or a
    #   String, optionally with an index like `"cuda:0"`) is returned as
    #   its String form without checking availability -- Torch itself
    #   raises a clear error the moment you actually try to use a device
    #   that isn't there, which is a better failure point than guessing
    #   here.
    # @returns [String] a device string, e.g. `"cpu"`, `"cuda"`, `"mps"`.
    def resolve(spec = :auto)
      return spec.to_s unless auto?(spec)

      if defined?(Torch::CUDA) && Torch::CUDA.available?
        "cuda"
      elsif defined?(Torch::Backends::MPS) && Torch::Backends::MPS.available?
        "mps"
      else
        "cpu"
      end
    end

    def auto?(spec)
      spec == :auto || spec == "auto"
    end
  end
end
