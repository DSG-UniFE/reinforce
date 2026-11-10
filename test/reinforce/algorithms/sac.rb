# frozen_string_literal: true

require "tmpdir"
require_relative "../../support/torch_test_helpers"
require "reinforce/algorithms/sac"
require_relative "../../support/agent_contract"

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

  include_context AgentContract, factory: -> { build_agent }, state: [0.0], actions: %i[bad good]

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

  describe "warmup" do
    # Counts the actions the agent takes, and the updates it makes.
    def recording_environment(taken)
      env = Object.new
      env.define_singleton_method(:state_size) { 1 }
      env.define_singleton_method(:actions) { %i[bad good] }
      env.define_singleton_method(:reset) { [0.0] }
      env.define_singleton_method(:step) do |action_index|
        taken << action_index
        [[0.0], 0.0, false]
      end
      env
    end

    # A policy that always picks :bad (index 0) with near certainty.
    def biased_policy
      layer = Torch::NN::Linear.new(1, 2)
      Torch::NN::Init.constant!(layer.weight, 0.0)
      layer.bias.data.copy!(Torch.tensor([20.0, -20.0]))
      Torch::NN::Sequential.new(layer)
    end

    it "acts uniformly at random and makes no updates during warmup, then follows the policy" do
      TorchTestHelpers.with_torch_seed do
        taken = []
        agent = Reinforce::Algorithms::SAC.new(
          recording_environment(taken),
          policy_model: biased_policy,
          q1_model: build_network(1, 2), q2_model: build_network(1, 2),
          q1_target_model: build_network(1, 2), q2_target_model: build_network(1, 2),
          batch_size: 4, warmup_steps: 40, auto_entropy_tuning: false, learning_rate: 0.0
        )

        agent.train(episodes: 1, steps_per_episode: 60)

        expect(taken.first(40).count(1)).to be > 10
        expect(taken.last(20).uniq).to be == [0]
        # Updates start on the step that completes the warmup.
        expect(agent.metrics[:loss].size).to be == 21
      end
    end

    it "counts warmup across #train calls rather than restarting it" do
      TorchTestHelpers.with_torch_seed do
        taken = []
        agent = Reinforce::Algorithms::SAC.new(
          recording_environment(taken),
          policy_model: biased_policy,
          q1_model: build_network(1, 2), q2_model: build_network(1, 2),
          q1_target_model: build_network(1, 2), q2_target_model: build_network(1, 2),
          batch_size: 4, warmup_steps: 40, auto_entropy_tuning: false, learning_rate: 0.0
        )

        agent.train(episodes: 1, steps_per_episode: 40)
        agent.train(episodes: 1, steps_per_episode: 20)

        expect(taken.last(20).uniq).to be == [0]
        expect(agent.metrics[:loss].size).to be == 21
      end
    end
  end

  it "targets 0.3 of the maximum entropy by default when tuning alpha" do
    agent = build_agent

    expect(agent.instance_variable_get(:@target_entropy)).to be_within(1e-12).of(0.3 * Math.log(2))
  end

  describe "reward normalization" do
    # A 1-dimensional environment rewarding 2.0 per step, terminating after
    # 3 steps.
    def three_step_environment
      env = Object.new
      steps = 0
      env.define_singleton_method(:state_size) { 1 }
      env.define_singleton_method(:actions) { %i[bad good] }
      env.define_singleton_method(:reset) { [(steps = 0).to_f] }
      env.define_singleton_method(:step) do |_action|
        steps += 1
        [[steps.to_f], 2.0, steps >= 3]
      end
      env
    end

    def build_normalizing_agent(**options)
      Reinforce::Algorithms::SAC.new(
        three_step_environment,
        policy_model: build_network(1, 2), q1_model: build_network(1, 2), q2_model: build_network(1, 2),
        q1_target_model: build_network(1, 2), q2_target_model: build_network(1, 2),
        discount_factor: 0.5, batch_size: 4, warmup_steps: 100, **options
      )
    end

    it "is on by default" do
      expect(build_normalizing_agent.config.normalize_rewards).to be == true
    end

    it "leaves rewards unscaled when disabled" do
      agent = build_normalizing_agent(normalize_rewards: false)
      agent.train(episodes: 2, steps_per_episode: 3)

      expect(agent.send(:reward_scale)).to be_within(1e-12).of(1.0)
    end

    it "scales by the running std of the per-episode discounted return, keeping raw rewards in the replay buffer" do
      agent = build_normalizing_agent
      agent.train(episodes: 2, steps_per_episode: 3)

      # Discounted returns (gamma 0.5), reset each episode: 2, 3, 3.5, twice.
      reference = Reinforce::RunningMeanStd.new
      [2.0, 3.0, 3.5, 2.0, 3.0, 3.5].each { |discounted_return| reference.update(discounted_return) }

      expect(agent.send(:reward_scale)).to be_within(1e-9).of(1.0 / Math.sqrt(reference.var + 1e-8))
      expect(agent.sample_batch(4).map { |transition| transition[:reward] }.uniq).to be == [2.0]
    end

    it "leaves rewards unscaled when transitions only arrive via #append_transition" do
      agent = build_normalizing_agent(normalize_rewards: true)
      4.times { agent.append_transition(state: [0.0], action: :good, reward: 5.0, next_state: [0.0], done: true) }

      expect(agent.send(:reward_scale)).to be_within(1e-12).of(1.0)
    end

    it "uses the scaled rewards in the critics' targets" do
      TorchTestHelpers.with_torch_seed do
        agent = build_normalizing_agent(normalize_rewards: true)
        agent.train(episodes: 2, steps_per_episode: 3)
        scale = agent.send(:reward_scale)
        batch = Array.new(4) { {state: [0.0], action_index: 1, reward: 2.0, next_state: [0.0], done: 1.0} }
        targets = []
        mse = agent.instance_variable_get(:@mse)
        mse.define_singleton_method(:call) do |prediction, target|
          targets << target.to_a
          super(prediction, target)
        end

        agent.train_step(batch:)

        # Terminal transitions: the target is exactly the scaled reward.
        expect(targets.first.all? { |target| (target - 2.0 * scale).abs < 1e-5 }).to be == true
        expect(scale).not.to be_within(1e-3).of(1.0)
      end
    end
  end
end
