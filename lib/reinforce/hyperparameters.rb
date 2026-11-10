# frozen_string_literal: true

module Reinforce
  # Every algorithm keeps its hyperparameters in a `Config` value object
  # (a `Data.define` with defaults, e.g. Reinforce::Algorithms::PPO::Config),
  # so a full hyperparameter set is one value that can be constructed,
  # logged (`agent.config.to_h`), compared and reused. Constructors take an
  # optional config plus individual keyword overrides:
  #
  #   PPO.new(environment, learning_rate: 1e-3)
  #   PPO.new(environment, PPO::Config.new(learning_rate: 1e-3))
  #   PPO.new(environment, base_config, ppo_epochs: 4) # base_config, one field changed
  #
  # Collaborators -- environments, datasets, models, optimizers -- are not
  # hyperparameters and stay ordinary constructor arguments.
  module Hyperparameters
    module_function

    # The config an algorithm runs with: `config` (or the class defaults)
    # with `overrides` applied. Unknown hyperparameter names raise
    # ArgumentError, so a typo can't silently fall back to a default.
    #
    # @parameter config_class [Class] the algorithm's Config.
    # @parameter config [Data, nil] a base config, or nil for the defaults.
    # @parameter overrides [Hash{Symbol => Object}] individual hyperparameters.
    # @returns [Data] an instance of `config_class`.
    def resolve(config_class, config, overrides)
      base = config || config_class.new
      unless base.is_a?(config_class)
        raise ArgumentError, "expected a #{config_class}, got #{base.class}"
      end

      overrides.empty? ? base : base.with(**overrides)
    end
  end
end
