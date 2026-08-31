# frozen_string_literal: true

require "tmpdir"
require_relative "../../support/torch_test_helpers"
require "reinforce/algorithms/sac"

describe Reinforce::Algorithms::SAC do
  let(:environment) do
    env = Object.new
    env.define_singleton_method(:state_size) { 1 }
    env.define_singleton_method(:actions) { %i[bad good] }
    env.define_singleton_method(:reset) { [0.0] }
    env.define_singleton_method(:step) do |action_index|
      reward = (action_index == 1) ? 1.0 : 0.0
      [[0.0], reward, true]
    end
    env
  end

  def build_network(input_size, output_size, hidden_size = 32)
    Torch::NN::Sequential.new(
      Torch::NN::Linear.new(input_size, hidden_size),
      Torch::NN::ReLU.new,
      Torch::NN::Linear.new(hidden_size, hidden_size),
      Torch::NN::ReLU.new,
      Torch::NN::Linear.new(hidden_size, output_size)
    )
  end

  def build_zero_network(input_size, output_size)
    layer = Torch::NN::Linear.new(input_size, output_size)
    Torch::NN::Init.constant!(layer.weight, 0.0)
    Torch::NN::Init.constant!(layer.bias, 0.0)
    Torch::NN::Sequential.new(layer)
  end

  def build_agent(**options)
    Reinforce::Algorithms::SAC.new(
      environment,
      policy_model: options.delete(:policy_model) || build_network(1, 2),
      q1_model: options.delete(:q1_model) || build_network(1, 2),
      q2_model: options.delete(:q2_model) || build_network(1, 2),
      q1_target_model: options.delete(:q1_target_model) || build_network(1, 2),
      q2_target_model: options.delete(:q2_target_model) || build_network(1, 2),
      learning_rate: 0.01,
      discount_factor: 0.9,
      entropy_coefficient: 0.05,
      auto_entropy_tuning: true,
      batch_size: 32,
      warmup_steps: 32,
      **options
    )
  end

  it "can train and accumulate losses/logs" do
    TorchTestHelpers.with_torch_seed do
      agent = build_agent(critic_updates_per_step: 1, policy_updates_per_step: 1, target_updates_per_step: 1)

      action_before = agent.predict([0.0])
      agent.train(episodes: 150, steps_per_episode: 1)
      action_after = agent.predict([0.0])

      expect(%i[bad good].include?(action_before)).to be == true
      expect(agent.logs[:episode_reward].size).to be == 150
      expect(agent.logs[:episode_length].size).to be == 150
      expect(agent.logs[:loss].size > 0).to be == true
      expect(agent.logs[:alpha].size > 0).to be == true
      expect(agent.logs[:entropy].size > 0).to be == true
      expect(action_after).to be == :good
    end
  end

  it "initializes target critics from the online critics and bounds replay capacity" do
    TorchTestHelpers.with_torch_seed do
      agent = build_agent(batch_size: 2, replay_size: 3)

      expect(TorchTestHelpers.parameter_snapshot(agent.q1_target_model)).to be == TorchTestHelpers.parameter_snapshot(agent.q1_model)
      expect(TorchTestHelpers.parameter_snapshot(agent.q2_target_model)).to be == TorchTestHelpers.parameter_snapshot(agent.q2_model)

      4.times do |index|
        agent.append_transition(
          state: [index.to_f], action: :good, reward: 1.0, next_state: [index.to_f], done: true
        )
      end

      expect(agent.replay_size).to be == 3
      expect(agent.ready_for_update?).to be == true
    end
  end

  it "does not bootstrap terminal transitions and updates all trainable models" do
    TorchTestHelpers.with_torch_seed do
      agent = build_agent(
        policy_model: build_zero_network(1, 2),
        q1_model: build_zero_network(1, 2),
        q2_model: build_zero_network(1, 2),
        q1_target_model: build_zero_network(1, 2),
        q2_target_model: build_zero_network(1, 2),
        batch_size: 1,
        tau: 0.5
      )
      batch = [{state: [0.0], action: :good, reward: 2.0, next_state: [100.0], done: 1.0}]
      q1_before = TorchTestHelpers.parameter_snapshot(agent.q1_model)
      policy_before = TorchTestHelpers.parameter_snapshot(agent.policy_model)
      target_before = TorchTestHelpers.parameter_snapshot(agent.q1_target_model)

      metrics = agent.train_step(batch:)

      expect(metrics[:q1_loss]).to be_within(1e-6).of(4.0)
      expect(metrics.values.all?(&:finite?)).to be == true
      expect(TorchTestHelpers.parameters_changed?(agent.q1_model, q1_before)).to be == true
      expect(TorchTestHelpers.parameters_changed?(agent.policy_model, policy_before)).to be == true
      expect(TorchTestHelpers.parameters_changed?(agent.q1_target_model, target_before)).to be == true
    end
  end

  it "round-trips model state through a checkpoint" do
    TorchTestHelpers.with_torch_seed do
      agent = build_agent(batch_size: 2, warmup_steps: 2)
      agent.train(episodes: 4, steps_per_episode: 1)
      prediction = agent.predict([0.0])

      Dir.mktmpdir do |directory|
        path = File.join(directory, "sac.pt")
        agent.save(path)
        restored = build_agent(batch_size: 2, warmup_steps: 2)
        restored.load(path)

        expect(TorchTestHelpers.parameter_snapshot(restored.policy_model)).to be == TorchTestHelpers.parameter_snapshot(agent.policy_model)
        expect(TorchTestHelpers.parameter_snapshot(restored.q1_model)).to be == TorchTestHelpers.parameter_snapshot(agent.q1_model)
        expect(restored.predict([0.0])).to be == prediction
      end
    end
  end
end
