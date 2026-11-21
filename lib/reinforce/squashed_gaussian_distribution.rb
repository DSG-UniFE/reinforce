# frozen_string_literal: true

require "torch"

# A diagonal Gaussian squashed by tanh, over continuous actions in [-1, 1]
# on each component (scaled onto a Reinforce::Spaces::Box by the agent).
#
# It works on *pre-tanh* values u: #sample and #choose return u, and
# #log_probability takes u, so an agent stores u and never inverts the
# tanh (atanh is unstable near ±1). The action is tanh(u) (see .squash),
# whose log-density is the Gaussian's minus the log-Jacobian of the tanh:
#
#   log π(a) = log N(u; mean, std²) − Σ_j log(1 − tanh(u_j)²)
#
# #entropy is the Gaussian's (over u), in closed form: the tanh-squashed
# distribution has none, and PPO's entropy bonus only needs to keep the
# spread from collapsing.
class SquashedGaussianDistribution
  LOG_SQRT_2PI = 0.5 * Math.log(2.0 * Math::PI)

  attr_reader :mean, :log_std

  # @parameter mean [Torch::Tensor] [components] or [batch, components].
  # @parameter log_std [Torch::Tensor] broadcastable to `mean` (e.g. a
  #   state-independent [components] parameter).
  def initialize(mean:, log_std:)
    @mean = mean
    @log_std = log_std.expand_as(mean)
  end

  # The action for pre-tanh values: tanh(u), in [-1, 1].
  def self.squash(pre_tanh) = Torch.tanh(pre_tanh)

  # @returns [Torch::Tensor] pre-tanh values u ~ N(mean, std²), shaped like the mean.
  def sample
    Torch.no_grad { @mean + (@log_std.exp * Torch.randn_like(@mean)) }
  end

  # @returns [Torch::Tensor] the mean, whose tanh is the deterministic action.
  def greedy = @mean

  # The rule shared by every algorithm's #predict.
  #
  # @returns [Array(Float)] pre-tanh values: the mean if deterministic, a
  #   sample otherwise.
  def choose(deterministic: true)
    (deterministic ? greedy : sample).reshape(-1).to_a
  end

  # @parameter pre_tanh [Torch::Tensor] values u shaped like the mean.
  # @returns [Torch::Tensor] log π(tanh(u)), summed over the components: a
  #   scalar, or one value per batch row.
  def log_probability(pre_tanh)
    pre_tanh = pre_tanh.to(@mean.dtype)
    gaussian = (((pre_tanh - @mean) / @log_std.exp)**2 * -0.5) - @log_std - LOG_SQRT_2PI
    # log(1 - tanh(u)^2), computed stably as 2 * (log 2 - u - softplus(-2u)).
    jacobian = (Torch::NN::Functional.softplus(pre_tanh * -2.0) + pre_tanh - Math.log(2.0)) * -2.0
    (gaussian - jacobian).sum(-1)
  end

  # @returns [Torch::Tensor] the entropy of the Gaussian over u, summed over
  #   the components: a scalar, or one value per batch row.
  def entropy
    (@log_std + 0.5 + LOG_SQRT_2PI).sum(-1)
  end
end
