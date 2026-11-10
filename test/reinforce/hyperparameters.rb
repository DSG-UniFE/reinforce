# frozen_string_literal: true

require "reinforce"
require_relative "../support/torch_test_helpers"

describe Reinforce::Hyperparameters do
  def environment
    env = Object.new
    env.define_singleton_method(:state_size) { 2 }
    env.define_singleton_method(:actions) { %i[left right] }
    env.define_singleton_method(:reset) { [0.0, 0.0] }
    env.define_singleton_method(:step) { |_action| [[0.0, 0.0], 0.0, true, {}] }
    env
  end

  def dataset
    data = Reinforce::OfflineDataset.new
    data.add(state: [0.0, 0.0], action: :left, reward: 0.0, next_state: [0.0, 0.0], done: true)
    data.add(state: [1.0, 0.0], action: :right, reward: 1.0, next_state: [1.0, 0.0], done: true)
    data
  end

  def network(output_size = 2)
    Torch::NN::Linear.new(2, output_size)
  end

  def sac_models
    {policy_model: network, q1_model: network, q2_model: network, q1_target_model: network, q2_target_model: network}
  end

  # For every configurable class: how to build one (from an optional base
  # config plus keyword overrides), and one hyperparameter to override.
  configurable = {
    Reinforce::QFunctionANN => [->(config, **hp) { Reinforce::QFunctionANN.new(2, 2, config, **hp) }, :discount_factor, 0.5],
    Reinforce::Algorithms::DQN => [->(config, **hp) { Reinforce::Algorithms::DQN.new(environment, config, **hp) }, :epsilon, 0.25],
    Reinforce::Algorithms::PPO => [->(config, **hp) { Reinforce::Algorithms::PPO.new(environment, config, **hp) }, :ppo_epochs, 3],
    Reinforce::Algorithms::SAC => [->(config, **hp) { Reinforce::Algorithms::SAC.new(environment, config, **sac_models, **hp) }, :warmup_steps, 7],
    Reinforce::Algorithms::SARSA => [
      ->(config, **hp) { Reinforce::Algorithms::SARSA.new(environment, Reinforce::QFunctionANN.new(2, 2), config, **hp) }, :epsilon, 0.25
    ],
    Reinforce::Algorithms::MonteCarloPolicyGradient => [
      lambda do |config, **hp|
        model = network
        Reinforce::Algorithms::MonteCarloPolicyGradient.new(environment, model, Torch::Optim::SGD.new(model.parameters, lr: 0.1), config, **hp)
      end,
      :entropy_coefficient, 0.5
    ],
    Reinforce::Algorithms::TemporalDifference => [
      ->(config, **hp) { Reinforce::Algorithms::TemporalDifference.new(environment, config, **hp) }, :epsilon, 0.25
    ],
    Reinforce::Algorithms::BehaviorCloning => [
      ->(config, **hp) { Reinforce::Algorithms::BehaviorCloning.new(dataset, config, **hp) }, :learning_rate, 0.5
    ],
    Reinforce::Algorithms::ConservativeQLearning => [
      ->(config, **hp) { Reinforce::Algorithms::ConservativeQLearning.new(dataset, config, **hp) }, :alpha, 0.5
    ],
    Reinforce::Algorithms::ExDM => [->(config, **hp) { Reinforce::Algorithms::ExDM.new(dataset, config, **hp) }, :noise_std, 0.5]
  }

  configurable.each do |klass, (build, field, value)|
    describe klass do
      it "runs with its Config's defaults when given no hyperparameters" do
        expect(instance_exec(nil, &build).config).to be == klass::Config.new
      end

      it "applies keyword overrides, leaving the other hyperparameters at their defaults" do
        config = instance_exec(nil, **{field => value}, &build).config

        expect(config.public_send(field)).to be == value
        expect(config.with(field => klass::Config.new.public_send(field))).to be == klass::Config.new
      end

      it "takes a base config, with keyword overrides applied on top" do
        base = klass::Config.new(field => value)

        expect(instance_exec(base, &build).config).to be_equal(base)
        overridden = instance_exec(base, **{field => klass::Config.new.public_send(field)}, &build).config
        expect(overridden).to be == klass::Config.new
      end

      it "rejects unknown hyperparameter names" do
        expect { instance_exec(nil, learning_rat: 0.1, &build) }.to raise_exception(ArgumentError, message: be =~ /learning_rat/)
      end

      it "rejects a config of the wrong class" do
        expect { instance_exec(Data.define(:x).new(x: 1), &build) }.to raise_exception(ArgumentError, message: be =~ /expected a/)
      end
    end
  end

  describe "hyperparameters taking effect" do
    it "configures PPO's optimizer, epochs and GAE lambda" do
      ppo = Reinforce::Algorithms::PPO.new(environment, learning_rate: 0.123, ppo_epochs: 3, gae_lambda: 0.5)

      expect(ppo.optimizer.param_groups[0][:lr]).to be_within(1e-9).of(0.123)
      expect(ppo.instance_variable_get(:@ppo_epochs)).to be == 3
      expect(ppo.instance_variable_get(:@gaelam)).to be_within(1e-12).of(0.5)
    end

    it "configures QFunctionANN's optimizer" do
      q_function = Reinforce::QFunctionANN.new(2, 2, learning_rate: 0.123)

      expect(q_function.optimizer.param_groups[0][:lr]).to be_within(1e-9).of(0.123)
    end

    it "configures DQN's update schedule and minibatch size" do
      dqn = Reinforce::Algorithms::DQN.new(environment, training_start: 0, q_update_frequency: 1, minibatch_size: 3)
      sampled_sizes = []
      replay = dqn.instance_variable_get(:@prioritized_experience_replay)
      replay.define_singleton_method(:sample) do |size|
        sampled_sizes << size
        super(size)
      end

      TorchTestHelpers.with_torch_seed { dqn.train(episodes: 2, steps_per_episode: 1) }

      expect(sampled_sizes).to be == [3, 3]
    end
  end
end
