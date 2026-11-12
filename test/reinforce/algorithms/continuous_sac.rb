# frozen_string_literal: true

require "tmpdir"
require "reinforce"
require_relative "../../support/torch_test_helpers"

# One-step task: the state is a number s in [-1, 1] and the reward is
# -(a - 2s)^2, so the optimal action is 2s (within the [-2, 2] action space).
class TrackingEnvironment
  include Reinforce::Environment

  attr_reader :actions_taken

  def initialize(truncate: false)
    @truncate = truncate
    @actions_taken = []
  end

  def state_size = 1

  def action_space = Reinforce::Spaces::Box.new(low: -2.0, high: 2.0, size: 1)

  def reset
    @state = [(rand * 2.0) - 1.0]
  end

  def step(action)
    @actions_taken << action
    reward = -((action[0] - (2.0 * @state[0]))**2)
    info = {metrics: {error: reward.abs}}
    info[:truncated] = true if @truncate
    [reset, reward, true, info]
  end
end

# Moves every action to 0.5 (in [-1, 1]) and reports how often it did so.
class ConstantShield
  attr_reader :calls, :loss_calls

  def initialize
    @calls = 0
    @loss_calls = 0
  end

  def call(_state, action)
    @calls += 1
    @last_info = {corrections: (action == [0.5]) ? 0 : 1}
    [0.5]
  end

  attr_reader :last_info

  # Pulls the actor's actions toward 0.5.
  def safety_loss(_states, actions)
    @loss_calls += 1
    ((actions - 0.5)**2).mean
  end
end

describe Reinforce::Algorithms::ContinuousSAC do
  let(:environment) { TrackingEnvironment.new }

  def build_agent(env = environment, **options)
    Reinforce::Algorithms::ContinuousSAC.new(env, warmup_steps: 64, batch_size: 32, hidden_size: 32, **options)
  end

  it "learns the optimal action of a one-step task" do
    TorchTestHelpers.with_torch_seed(1) do
      agent = build_agent(warmup_steps: 200, batch_size: 64, hidden_size: 64)
      agent.train(episodes: 1500, steps_per_episode: 1)

      [-0.5, 0.0, 0.5].each do |state|
        expect(agent.predict([state])[0]).to be_within(0.5).of(2.0 * state)
      end
      expect(agent.metrics.mean(:episode_reward, window: 200)).to be > -0.5
    end
  end

  it "records losses, entropy, alpha and the environment's metrics" do
    TorchTestHelpers.with_torch_seed do
      agent = build_agent
      agent.train(episodes: 100, steps_per_episode: 1)

      expect(agent.metrics[:episode_reward].size).to be == 100
      expect(agent.metrics[:episode_length].size).to be == 100
      expect(agent.metrics[:episode_error].size).to be == 100
      # updates start once the warmup is over
      expect(agent.metrics[:loss].size).to be == 100 - 64 + 1
      %i[q1_loss q2_loss policy_loss alpha_loss entropy alpha].each do |key|
        expect(agent.metrics[key].all?(&:finite?)).to be == true
      end
      expect(agent.metrics.key?(:safety_loss)).to be == false
      expect(agent.replay_size).to be == 100
    end
  end

  it "predicts elements of the action space, greedily by default" do
    TorchTestHelpers.with_torch_seed do
      agent = build_agent
      greedy = [1, 2, 3].map { |seed| TorchTestHelpers.with_torch_seed(seed) { agent.predict([0.3]) } }
      sampled = Array.new(20) { agent.predict([0.3], deterministic: false) }

      expect(greedy.uniq.size).to be == 1
      expect(sampled.uniq.size > 1).to be == true
      expect((greedy + sampled).all? { |action| agent.action_space.contains?(action) }).to be == true
    end
  end

  it "keeps a fixed entropy coefficient without automatic tuning" do
    TorchTestHelpers.with_torch_seed do
      agent = build_agent(auto_entropy_tuning: false, entropy_coefficient: 0.2)
      agent.train(episodes: 80, steps_per_episode: 1)

      expect(agent.alpha).to be_equal(0.2)
      expect(agent.metrics[:alpha_loss].uniq).to be == [0.0]
    end
  end

  it "bootstraps across truncated episodes" do
    TorchTestHelpers.with_torch_seed do
      agent = build_agent(TrackingEnvironment.new(truncate: true))
      agent.train(episodes: 10, steps_per_episode: 1)
      _states, _actions, _rewards, _next_states, dones = agent.instance_variable_get(:@replay).sample(10)

      expect(dones.to_a.uniq).to be == [0.0]
    end
  end

  it "executes shielded actions, stores base actions and adds the safety loss" do
    TorchTestHelpers.with_torch_seed do
      shield = ConstantShield.new
      agent = build_agent(shield: shield, safety_weight: 1.0)
      agent.train(episodes: 100, steps_per_episode: 1)
      _states, actions, _rewards, _next_states, _dones = agent.instance_variable_get(:@replay).sample(32)

      # 0.5 in [-1, 1] is 1.0 in the [-2, 2] action space
      expect(environment.actions_taken.uniq).to be == [[1.0]]
      expect(actions.to_a.flatten.uniq.size > 1).to be == true
      expect(shield.calls).to be == 100
      expect(shield.loss_calls).to be == agent.metrics[:loss].size
      expect(agent.metrics[:safety_loss].all?(&:finite?)).to be == true
      expect(agent.metrics[:shield_corrections].sum).to be == 100
      expect(agent.predict([0.0])).to be == [1.0]
    end
  end

  it "saves and loads its networks" do
    TorchTestHelpers.with_torch_seed do
      agent = build_agent
      agent.train(episodes: 80, steps_per_episode: 1)
      other = build_agent

      Dir.mktmpdir do |directory|
        path = File.join(directory, "sac.pth")
        agent.save(path)
        other.load(path)
      end

      expect(other.predict([0.4])).to be == agent.predict([0.4])
      expect(TorchTestHelpers.parameter_snapshot(other.q2_target_model)).to be == TorchTestHelpers.parameter_snapshot(agent.q2_target_model)
    end
  end

  it "accepts custom networks" do
    TorchTestHelpers.with_torch_seed do
      critic = -> { Reinforce::Networks.mlp(2, 1, hidden_size: 8) }
      q1 = critic.call
      q1_target = critic.call
      agent = build_agent(
        policy_model: Reinforce::Networks.mlp(1, 2, hidden_size: 8),
        q1_model: q1, q2_model: critic.call, q1_target_model: q1_target, q2_target_model: critic.call
      )

      expect(agent.q1_target_model).to be_equal(q1_target)
      expect(TorchTestHelpers.parameter_snapshot(q1_target)).to be == TorchTestHelpers.parameter_snapshot(q1)
      expect { build_agent(q1_model: critic.call) }.to raise_exception(ArgumentError, message: be =~ /target models/)
    end
  end

  it "requires a continuous action space" do
    discrete = Object.new
    discrete.define_singleton_method(:action_space) { %i[left right] }
    discrete.define_singleton_method(:state_size) { 1 }

    expect { build_agent(discrete) }.to raise_exception(ArgumentError, message: be =~ /Box/)
  end
end
