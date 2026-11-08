# frozen_string_literal: true

require "reinforce/device"

describe Reinforce::Device do
  describe ".resolve" do
    it "returns cpu when no accelerator is available (this test's expected CI/dev environment)" do
      # This assumes the machine running the suite has no CUDA/MPS, which
      # is true in CI and matches how this whole library is currently
      # CPU-only in practice. If this ever runs somewhere with a GPU,
      # this specific assertion would need a stub -- but that's a real
      # environment difference worth knowing about, not a flaky test.
      expect(Reinforce::Device.resolve(:auto)).to be == "cpu" unless Torch::CUDA.available? || Torch::Backends::MPS.available?
    end

    it "passes through an explicit device spec without checking availability" do
      expect(Reinforce::Device.resolve(:cpu)).to be == "cpu"
      expect(Reinforce::Device.resolve("cuda:0")).to be == "cuda:0"
    end

    it "defaults to :auto when called with no argument" do
      expect(Reinforce::Device.resolve).to be == Reinforce::Device.resolve(:auto)
    end
  end
end
