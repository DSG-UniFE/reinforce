# frozen_string_literal: true

require "torch"
require_relative "../hyperparameters"
require_relative "../networks"

module Reinforce
  module IntrinsicRewards
    # Random Network Distillation (Burda et al., 2018): a fixed, randomly
    # initialized *target* network maps each state to a feature vector, and
    # a *predictor* network is trained to reproduce those features on the
    # states the agent visits. The bonus is the predictor's error: low on
    # states like the ones it has been trained on, high on novel ones. Unlike
    # CountBased it needs no exact state matching, so it generalizes to large
    # or continuous state spaces -- novelty is judged by similarity, the way
    # the networks happen to generalize.
    #
    # Observations are normalized by a running per-dimension mean and
    # standard deviation and clipped to [-observation_clip, observation_clip]
    # before both networks (as in the paper, where RND barely worked
    # without it). #observe updates those statistics; #train fits the
    # predictor on the states it is given; #call returns
    # coefficient * mean squared prediction error per state.
    class RND
      # Hyperparameters (see Reinforce::Hyperparameters).
      #
      # coefficient: scale of the bonus. (PPO's separate intrinsic value
      #   head normalizes the intrinsic stream, which cancels it out.)
      # learning_rate / update_epochs: predictor optimization, per #train.
      # feature_size: width of the target's output.
      # network: shape of both networks' hidden layers.
      # observation_clip: bound on normalized observations.
      Config = Data.define(:coefficient, :learning_rate, :update_epochs, :feature_size, :network, :observation_clip) do
        def initialize(coefficient: 1.0, learning_rate: 1e-3, update_epochs: 4, feature_size: 64,
          network: Reinforce::Networks::Config.new, observation_clip: 5.0)
          super
        end
      end

      attr_reader :config, :target, :predictor

      def initialize(state_size, config = nil, **hyperparameters)
        @config = Reinforce::Hyperparameters.resolve(Config, config, hyperparameters)
        @target = @config.network.build(state_size, @config.feature_size)
        @target.parameters.each { |parameter| parameter.requires_grad = false }
        @predictor = @config.network.build(state_size, @config.feature_size)
        @optimizer = Torch::Optim::Adam.new(@predictor.parameters, lr: @config.learning_rate)
        @observation_mean = Torch.zeros(state_size)
        @observation_var = Torch.ones(state_size)
        @observation_count = 1e-4
      end

      # Updates the running observation statistics with one state.
      def observe(state)
        x = as_tensor([state])[0]
        delta = x - @observation_mean
        total = @observation_count + 1
        @observation_mean += delta / total
        # Parallel-variance update with a batch of one (see RunningMeanStd).
        @observation_var = (@observation_var * @observation_count + delta.pow(2) * @observation_count / total) / total
        @observation_count = total
        self
      end

      def call(states)
        return [] if states.empty?

        Torch.no_grad { prediction_errors(states) }.to_a.map { |error| @config.coefficient * error }
      end

      # Fits the predictor to the target on `states` (update_epochs full-batch
      # gradient steps).
      def train(states)
        return self if states.empty?

        @config.update_epochs.times do
          loss = prediction_errors(states).mean
          @optimizer.zero_grad
          loss.backward
          @optimizer.step
        end
        self
      end

      private

      def prediction_errors(states)
        inputs = normalize(as_tensor(states))
        (@predictor.call(inputs) - @target.call(inputs)).pow(2).mean(1)
      end

      def normalize(inputs)
        std = (@observation_var + 1e-8).sqrt
        ((inputs - @observation_mean) / std).clamp(-@config.observation_clip, @config.observation_clip)
      end

      def as_tensor(states)
        return states.to(:float32) if states.is_a?(Torch::Tensor)

        rows = states.map { |state| state.is_a?(Torch::Tensor) ? state.to_a : state }
        Torch.tensor(rows.map { |row| row.map(&:to_f) }, dtype: :float32)
      end
    end
  end
end
