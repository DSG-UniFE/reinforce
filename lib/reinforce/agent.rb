# frozen_string_literal: true

module Reinforce
  # The one thing every algorithm in this library has in common: given a
  # state, it can produce an action. Including this module documents that,
  # and turns a missing #predict into a clear NotImplementedError.
  #
  # Deliberately not part of this contract: #train, #save, and #load. They
  # are common (most algorithms have all three) but not universal --
  # TemporalDifference exposes a single-step #learn instead of a #train
  # loop of its own, and offline algorithms like ExDM have no tabular
  # state or #save/#load pair to persist yet. The online algorithms
  # (SARSA, DQN, PPO, SAC, MonteCarloPolicyGradient) share one
  # standardized signature,
  # `train(episodes:, steps_per_episode:, **kwargs)`, plus a `logs`
  # reader, so generic driver and benchmarking code can train any of them
  # the same way; offline ones generally expose
  # `train(epochs:, batch_size:)`. Forcing both shapes
  # into one required signature would paper over a real difference
  # (online vs. offline training) rather than remove duplication, so
  # only the online family was unified.
  module Agent
    # Every implementation follows the same two rules, so evaluation and
    # benchmarking code can drive any algorithm through this one method:
    #
    # 1. It returns an element of the environment's `actions` array (e.g.
    #    `:up`), never an index or a Tensor. Environments' #step accept
    #    that directly.
    # 2. It is greedy by default (`deterministic: true`), because that is
    #    what evaluating a learned policy wants. With `deterministic:
    #    false` it samples instead: from the policy distribution for
    #    policy-based algorithms, and from a softmax over Q-values
    #    (Boltzmann exploration) for value-based ones.
    #
    # @parameter state [Object] a state from the environment's state space.
    # @parameter deterministic [Boolean] greedy if true, sampled if false.
    # @returns [Object] an element of the environment's action space.
    def predict(state, deterministic: true)
      raise NotImplementedError, "#{self.class} must implement #predict"
    end
  end
end
