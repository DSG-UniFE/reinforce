# frozen_string_literal: true

require_relative "../hyperparameters"

module Reinforce
  module IntrinsicRewards
    # Count-based exploration bonus: coefficient / sqrt(N(s)), where N(s) is
    # how many times the agent has reached state s over its whole training
    # (the MBIE-EB form, Strehl & Littman 2008; see Bellemare et al. 2016
    # for its pseudo-count generalization to large state spaces). Rarely
    # visited states pay more, so the agent is drawn to the edge of what it
    # has explored; the bonus decays as everything becomes familiar.
    #
    # Counts are exact, so this suits small discrete state spaces (GridWorld,
    # Taxi). For continuous states, pass a `key:` that discretizes them.
    class CountBased
      # Hyperparameters (see Reinforce::Hyperparameters).
      #
      # coefficient: the bonus for a state seen exactly once.
      Config = Data.define(:coefficient) do
        def initialize(coefficient: 0.1)
          super
        end
      end

      attr_reader :config

      # @parameter key [#call] maps a state to the value counts are kept by;
      #   defaults to the state's Array form.
      def initialize(config = nil, key: nil, **hyperparameters)
        @config = Reinforce::Hyperparameters.resolve(Config, config, hyperparameters)
        @key = key || ->(state) { state.respond_to?(:to_a) ? state.to_a : state }
        @counts = Hash.new(0)
      end

      def observe(state)
        @counts[@key.call(state)] += 1
        self
      end

      # A state never observed gets the full coefficient, as if seen once.
      def call(states)
        states.map { |state| @config.coefficient / Math.sqrt([count(state), 1].max) }
      end

      def train(_states)
        self
      end

      # @returns [Integer] how many times `state` has been observed.
      def count(state)
        @counts[@key.call(state)]
      end
    end
  end
end
