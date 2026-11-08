# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi.

require "tmpdir"
require "reinforce/algorithms/monte_carlo_policy_gradient"
require_relative "../../support/torch_test_helpers"
require_relative "../../support/agent_contract"

# A two-dimensional, three-action environment that ends an episode after
# `episode_length` steps, so tests can check #train's episode bookkeeping.
class ThreeActionEnv
  def initialize(episode_length: 3)
    @episode_length = episode_length
  end

  def state_size
    2
  end

  def actions
    %i[a b c]
  end

  def reset
    @steps = 0
    [0.5, -0.5]
  end

  def step(_action)
    @steps += 1
    [[0.5, -0.5], 1.0, @steps >= @episode_length]
  end
end

describe Reinforce::Algorithms::MonteCarloPolicyGradient do
  let(:environment) { ThreeActionEnv.new }
  let(:policy) { Torch::NN::Linear.new(2, 3) }
  let(:sgd) { Torch::Optim::SGD.new(policy.parameters, lr: 0.1) }
  let(:agent) { Reinforce::Algorithms::MonteCarloPolicyGradient.new(environment, policy, sgd) }

  include_context AgentContract, factory: -> { agent }, state: [0.5, -0.5], actions: %i[a b c]

  it "chooses an Integer action from a plain Array state" do
    # Regression: #choose_action used to pass CategoricalDistribution an
    # Array (logits.to_a) instead of a Tensor, crashing on the first step,
    # and returned a 0-dim tensor that environments' #step can't map onto
    # an action.
    TorchTestHelpers.with_torch_seed do
      action = agent.choose_action([0.5, -0.5])

      expect(action.is_a?(Integer)).to be == true
      expect((0...3).cover?(action)).to be == true
    end
  end

  it "predicts without recording into the training buffer" do
    5.times { agent.predict([0.5, -0.5]) }

    expect(agent.instance_variable_get(:@log_probs)).to be == []
  end

  it "updates the policy parameters at the end of an episode" do
    # Regression: #update_policy never called loss.backward, so the
    # optimizer found no gradients and the policy never changed.
    TorchTestHelpers.with_torch_seed do
      snapshot = TorchTestHelpers.parameter_snapshot(policy)
      3.times do
        agent.choose_action([0.5, -0.5])
        agent.get_reward(1.0)
      end
      agent.update_policy

      expect(TorchTestHelpers.parameters_changed?(policy, snapshot)).to be == true
    end
  end

  it "clears the episode buffers so consecutive episodes can each be learned from" do
    # Regression: the buffers were never cleared, so a second episode's
    # update would try to backpropagate through the first episode's
    # already-freed graph.
    TorchTestHelpers.with_torch_seed do
      2.times do
        agent.choose_action([0.5, -0.5])
        agent.get_reward(1.0)
        agent.update_policy

        expect(agent.instance_variable_get(:@log_probs)).to be == []
        expect(agent.instance_variable_get(:@entropies)).to be == []
        expect(agent.instance_variable_get(:@rewards)).to be == []
      end
    end
  end

  describe "#train" do
    it "runs each episode until done, updating the policy once per episode" do
      TorchTestHelpers.with_torch_seed do
        snapshot = TorchTestHelpers.parameter_snapshot(policy)
        agent.train(episodes: 4, steps_per_episode: 10)

        expect(agent.logs[:loss].size).to be == 4
        expect(agent.logs[:episode_length]).to be == [3, 3, 3, 3]
        expect(agent.logs[:episode_reward]).to be == [3.0, 3.0, 3.0, 3.0]
        expect(agent.instance_variable_get(:@log_probs)).to be == []
        expect(TorchTestHelpers.parameters_changed?(policy, snapshot)).to be == true
      end
    end

    it "cuts episodes off at steps_per_episode" do
      TorchTestHelpers.with_torch_seed do
        agent.train(episodes: 2, steps_per_episode: 2)

        expect(agent.logs[:episode_length]).to be == [2, 2]
      end
    end

    it "skips the policy update for an episode with no steps" do
      # Torch.stack raises on an empty list, so a zero-step episode must not
      # reach #update_policy.
      agent.train(episodes: 1, steps_per_episode: 0)

      expect(agent.logs[:loss]).to be == []
      expect(agent.logs[:episode_length]).to be == [0]
    end
  end

  describe "return normalization" do
    # Runs one 3-step episode with every reward multiplied by `scale`, from a
    # fixed initial policy and a fixed seed (so the sampled actions match
    # across calls), and returns the updated policy's parameters.
    def parameters_after_episode(scale, **options)
      TorchTestHelpers.with_torch_seed do
        scaled_policy = Torch::NN::Linear.new(2, 3)
        optimizer = Torch::Optim::SGD.new(scaled_policy.parameters, lr: 0.1)
        scaled_agent = Reinforce::Algorithms::MonteCarloPolicyGradient.new(environment, scaled_policy, optimizer, **options)
        [1.0, -2.0, 3.0].each do |reward|
          scaled_agent.choose_action([0.5, -0.5])
          scaled_agent.get_reward(reward * scale)
        end
        scaled_agent.update_policy
        TorchTestHelpers.parameter_snapshot(scaled_policy).flatten
      end
    end

    it "makes the update independent of the reward scale" do
      # Regression: unnormalized returns in the thousands (GridWorld) made
      # updates large enough to collapse the policy onto one fixed path.
      small = parameters_after_episode(1.0)
      large = parameters_after_episode(1000.0)

      small.zip(large).each { |a, b| expect(a).to be_within(1e-4).of(b) }
    end

    it "can be turned off, leaving the update proportional to the raw returns" do
      small = parameters_after_episode(1.0, normalize_returns: false)
      large = parameters_after_episode(1000.0, normalize_returns: false)

      expect(small.zip(large).any? { |a, b| (a - b).abs > 1e-2 }).to be == true
    end
  end

  describe "entropy bonus" do
    # With all-zero rewards (and normalization off, so they stay zero), the
    # policy-gradient term vanishes and the entropy bonus is the only thing
    # the update can respond to.
    def zero_reward_update(entropy_coefficient)
      TorchTestHelpers.with_torch_seed do
        bonus_policy = Torch::NN::Linear.new(2, 3)
        optimizer = Torch::Optim::SGD.new(bonus_policy.parameters, lr: 0.5)
        bonus_agent = Reinforce::Algorithms::MonteCarloPolicyGradient.new(
          environment, bonus_policy, optimizer, normalize_returns: false, entropy_coefficient: entropy_coefficient
        )
        entropy = -> { CategoricalDistribution.new(logits: Torch.no_grad { bonus_policy.call(Torch.tensor([0.5, -0.5])) }).entropy.item }
        snapshot = TorchTestHelpers.parameter_snapshot(bonus_policy)
        before = entropy.call
        3.times do
          bonus_agent.choose_action([0.5, -0.5])
          bonus_agent.get_reward(0.0)
        end
        bonus_agent.update_policy
        {before: before, after: entropy.call, changed: TorchTestHelpers.parameters_changed?(bonus_policy, snapshot)}
      end
    end

    it "pushes the policy toward higher entropy" do
      result = zero_reward_update(0.1)

      expect(result[:after] > result[:before]).to be == true
    end

    it "can be turned off" do
      expect(zero_reward_update(0.0)[:changed]).to be == false
    end

    it "logs the mean policy entropy of each trained episode" do
      agent.train(episodes: 3, steps_per_episode: 10)

      expect(agent.logs[:entropy].size).to be == 3
      # Three actions: entropy lies between 0 and ln(3).
      expect(agent.logs[:entropy].all? { |e| e > 0 && e <= Math.log(3) + 1e-6 }).to be == true
    end
  end

  it "saves and loads the policy network's weights" do
    # Regression: #save/#load called @model.save/@model.load, which a plain
    # Torch::NN::Module doesn't define.
    TorchTestHelpers.with_torch_seed do
      Dir.mktmpdir do |dir|
        path = File.join(dir, "policy.pth")
        agent.save(path)

        other_policy = Torch::NN::Linear.new(2, 3)
        restored = Reinforce::Algorithms::MonteCarloPolicyGradient.new(environment, other_policy, Torch::Optim::SGD.new(other_policy.parameters, lr: 0.1))
        restored.load(path)

        expect(TorchTestHelpers.parameter_snapshot(other_policy)).to be == TorchTestHelpers.parameter_snapshot(policy)
      end
    end
  end
end
