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
  # state or #save/#load pair to persist yet. Online algorithms generally
  # expose `train(num_episodes, steps_per_episode)` plus a `logs` reader;
  # offline ones generally expose `train(epochs:, batch_size:)`. Forcing
  # both shapes into one required signature would paper over a real
  # difference (online vs. offline training) rather than remove
  # duplication, so it's left undone here -- see roadmap.md for the fuller
  # discussion of what a shared training entry point would need to look
  # like.
  module Agent
    # @parameter state [Object] a state from the environment's state space.
    # @returns [Object] an action from the environment's action space.
    def predict(state)
      raise NotImplementedError, "#{self.class} must implement #predict"
    end
  end
end
