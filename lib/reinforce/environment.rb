# frozen_string_literal: true

module Reinforce
  # The contract every environment in this library follows. Environments
  # are plain duck-typed Ruby objects -- nothing here is required at
  # runtime by the algorithms, which just call #reset/#step/etc directly --
  # but including this module in a new environment documents the contract
  # in one place and turns a missing method into a clear NotImplementedError
  # instead of a confusing NoMethodError raised from deep inside whichever
  # algorithm happened to call it first.
  #
  # See readme.md's "Define a new environment" section for the narrative
  # version of this contract, and test/support/environment_contract.rb for
  # a shared test context that checks a concrete environment actually
  # satisfies it.
  module Environment
    # Reset the environment to its initial state.
    #
    # @returns [Object] the initial state.
    def reset
      raise NotImplementedError, "#{self.class} must implement #reset"
    end

    # @returns [Integer] the size of the state vector.
    def state_size
      raise NotImplementedError, "#{self.class} must implement #state_size"
    end

    # @returns [Array] the available actions. For a discrete action space,
    #   `actions.size` gives the number of actions, and #step accepts
    #   either an element of this array or its index into it.
    def actions
      raise NotImplementedError, "#{self.class} must implement #actions"
    end

    # The action space of an environment with continuous actions (see
    # Reinforce::Algorithms::ContinuousSAC) or multi-discrete ones (see
    # Reinforce::Algorithms::PPO), which implements this instead of
    # #actions.
    #
    # @returns [Reinforce::Spaces::Box, Reinforce::Spaces::MultiDiscrete]
    def action_space
      raise NotImplementedError, "#{self.class} must implement #action_space for continuous actions"
    end

    # Which actions are allowed in the current state. Optional: the default
    # allows every action. Algorithms that support masks (PPO, DQN) never
    # choose a masked action, in training or in #predict.
    #
    # @returns [Array(Boolean), Array(Array(Boolean)), nil] one boolean per
    #   action (true = allowed); for a multi-discrete #action_space, one
    #   such Array per dimension; nil for no mask.
    def action_mask
      nil
    end

    # Execute an action and advance the environment by one step.
    #
    # @parameter action [Object] an element of #actions, or its index (an
    #   element of #action_space for continuous actions).
    # @returns [Array(Object, Numeric, Boolean, Hash)] a 4-tuple of
    #   `(next_state, reward, done, info)`. `info` carries auxiliary
    #   diagnostic data and is `{}` when there is none -- no algorithm in
    #   this library depends on it being non-empty.
    def step(action)
      raise NotImplementedError, "#{self.class} must implement #step"
    end

    # Render the environment to an output stream. Optional: unlike the
    # methods above, no algorithm in this library depends on #render, so
    # the default implementation is a no-op rather than a raise.
    #
    # @parameter output_stream [IO] where to render to.
    def render(output_stream = $stdout)
    end
  end
end
