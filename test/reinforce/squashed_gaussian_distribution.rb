# frozen_string_literal: true

require "reinforce"
require_relative "../support/torch_test_helpers"

describe SquashedGaussianDistribution do
  def reference_log_probability(u, mean, log_std)
    std = Math.exp(log_std)
    gaussian = (-0.5 * (((u - mean) / std)**2)) - log_std - (0.5 * Math.log(2.0 * Math::PI))
    gaussian - Math.log(1.0 - (Math.tanh(u)**2))
  end

  it "gives the log-density of tanh(u), Gaussian minus the tanh's log-Jacobian" do
    mean = Torch.tensor([[0.3, -1.0], [0.0, 2.0]])
    log_std = Torch.tensor([-0.5, 0.2])
    u = Torch.tensor([[0.1, -1.5], [-0.7, 2.5]])
    values = SquashedGaussianDistribution.new(mean: mean, log_std: log_std).log_probability(u).to_a

    expected = u.to_a.each_with_index.map do |row, b|
      row.each_with_index.sum { |value, j| reference_log_probability(value, mean.to_a[b][j], log_std.to_a[j]) }
    end
    expected.each_with_index { |value, b| expect(values[b]).to be_within(1e-4).of(value) }
  end

  it "is a probability density over actions in (-1, 1)" do
    distribution = SquashedGaussianDistribution.new(mean: Torch.tensor([0.3]), log_std: Torch.tensor([-0.5]))
    points = 20_000
    width = 2.0 / points
    actions = Array.new(points) { |i| -1.0 + ((i + 0.5) * width) }
    pre_tanh = Torch.tensor(actions.map { |a| [Math.atanh(a)] })
    densities = SquashedGaussianDistribution.new(mean: distribution.mean.expand(points, 1), log_std: distribution.log_std[0].reshape(1))
      .log_probability(pre_tanh).exp.to_a

    expect(densities.sum * width).to be_within(1e-3).of(1.0)
  end

  it "has the Gaussian's entropy over u, per batch row" do
    log_std = Torch.tensor([-0.5, 0.2])
    entropy = SquashedGaussianDistribution.new(mean: Torch.zeros(3, 2), log_std: log_std).entropy.to_a
    expected = log_std.to_a.sum { |l| 0.5 * Math.log(2.0 * Math::PI * Math::E * Math.exp(2.0 * l)) }

    expect(entropy.size).to be == 3
    entropy.each { |value| expect(value).to be_within(1e-5).of(expected) }
  end

  it "samples pre-tanh values around the mean, and chooses the mean when deterministic" do
    TorchTestHelpers.with_torch_seed do
      distribution = SquashedGaussianDistribution.new(mean: Torch.tensor([1.5, -0.5]), log_std: Torch.tensor([Math.log(0.5), Math.log(2.0)]))
      samples = Torch.stack(Array.new(4000) { distribution.sample })

      expect(samples.mean(0).to_a[0]).to be_within(0.05).of(1.5)
      expect(samples.mean(0).to_a[1]).to be_within(0.15).of(-0.5)
      expect(samples.std(0).to_a[0]).to be_within(0.05).of(0.5)
      expect(samples.std(0).to_a[1]).to be_within(0.15).of(2.0)
      # beyond the tanh's useful range, the action saturates but u doesn't
      expect(distribution.choose(deterministic: true)).to be == [1.5, -0.5]
    end
  end

  it "is differentiable in the mean and the log standard deviation" do
    mean = Torch.tensor([0.2, -0.3]).requires_grad!(true)
    log_std = Torch.tensor([0.0, -1.0]).requires_grad!(true)
    distribution = SquashedGaussianDistribution.new(mean: mean, log_std: log_std)
    (distribution.log_probability(Torch.tensor([0.5, 0.1])) + distribution.entropy).backward

    expect(mean.grad.abs.sum.item).to be > 0.0
    expect(log_std.grad.abs.sum.item).to be > 0.0
  end

  it "stays finite where the tanh saturates" do
    distribution = SquashedGaussianDistribution.new(mean: Torch.tensor([0.0]), log_std: Torch.tensor([2.0]))

    expect(TorchTestHelpers.finite_tensor?(distribution.log_probability(Torch.tensor([30.0])))).to be == true
  end
end
