# frozen_string_literal: true

require "tmpdir"
require "reinforce"
require_relative "../../support/torch_test_helpers"

# One-step task with continuous actions in [low, high]: the state is s in
# [-1, 1], and the reward is -(a - target(s))^2 with target(s) = mid + s *
# half-width / 2, so the best action is inside the box. Records every action.
class ContinuousTarget
  include Reinforce::Environment

  attr_reader :actions_taken

  def initialize(low: -1.0, high: 1.0)
    @low = low
    @high = high
    @actions_taken = []
  end

  def state_size = 1

  def action_space = Reinforce::Spaces::Box.new(low: @low, high: @high, size: 1)

  def target(state) = ((@low + @high) / 2.0) + (state * (@high - @low) / 4.0)

  def reset
    @state = [(rand * 2.0) - 1.0]
  end

  def step(action)
    @actions_taken << action
    reward = -((action[0] - target(@state[0]))**2)
    [reset, reward, true, {}]
  end
end

describe Reinforce::Algorithms::PPO do
  with "a continuous (Box) action space" do
    it "uses a squashed Gaussian policy with a learnt standard deviation" do
      TorchTestHelpers.with_torch_seed do
        agent = subject.new(ContinuousTarget.new, initial_log_std: -0.5)

        expect(agent.agent.log_std.to_a).to be == [-0.5]
        expect(agent.trainable_parameters.any? { |parameter| parameter.equal?(agent.agent.log_std) }).to be == true
        expect(agent.agent.distribution(Torch.zeros(1))).to be_a(SquashedGaussianDistribution)
      end
    end

    it "sends the environment actions scaled onto the box" do
      TorchTestHelpers.with_torch_seed do
        environment = ContinuousTarget.new(low: 0.0, high: 10.0)
        agent = subject.new(environment, initial_log_std: 1.0, ppo_epochs: 1, minibatch_size: 16)
        agent.train(episodes: 4, steps_per_episode: 32)

        expect(environment.actions_taken.size).to be == 128
        expect(environment.actions_taken.all? { |action| action.size == 1 && action[0].between?(0.0, 10.0) }).to be == true
        # a wide initial spread reaches both halves of the box
        expect(environment.actions_taken.map(&:first).minmax.then { |low, high| low < 3.0 && high > 7.0 }).to be == true
        expect(agent.predict([0.2])[0].between?(0.0, 10.0)).to be == true
      end
    end

    it "learns the best action of each state" do
      TorchTestHelpers.with_torch_seed do
        environment = ContinuousTarget.new(low: -2.0, high: 2.0)
        agent = subject.new(environment, learning_rate: 3e-3, ppo_epochs: 4, minibatch_size: 32, discount_factor: 0.0,
          entropy_coefficient: 0.0, max_grad_norm: 0.5)
        states = [-0.8, -0.3, 0.0, 0.4, 0.9]
        error = -> { states.sum { |s| (agent.predict([s])[0] - environment.target(s)).abs } / states.size }
        before = error.call
        agent.train(episodes: 150, steps_per_episode: 64)

        expect(error.call).to be < 0.1
        expect(error.call).to be < before / 2
      end
    end

    it "saves and loads the standard deviation with the networks" do
      TorchTestHelpers.with_torch_seed do
        agent = subject.new(ContinuousTarget.new, initial_log_std: -0.3)
        agent.agent.log_std.data.fill!(0.7)
        Dir.mktmpdir do |directory|
          path = File.join(directory, "ppo.pt")
          agent.save(path)
          restored = subject.new(ContinuousTarget.new)
          restored.load(path)

          expect(restored.agent.log_std.to_a.first).to be_within(1e-6).of(0.7)
          expect(restored.predict([0.5])).to be == agent.predict([0.5])
        end
      end
    end
  end

  with "a shield" do
    # Executes a fixed action, records what it was given, and has a
    # differentiable safety loss pulling the policy's action toward -1.
    let(:shield) do
      Class.new do
        attr_reader :calls, :loss_calls, :shapes, :last_info

        def initialize
          @calls = []
          @loss_calls = 0
          @shapes = []
        end

        def call(state, action)
          @calls << [state, action]
          @last_info = {interventions: 1, safety: 0.5}
          [0.5]
        end

        def safety_loss(states, actions)
          @loss_calls += 1
          @shapes << actions.shape
          raise "actions must be in [-1, 1]" unless actions.abs.max.item <= 1.0

          ((actions + 1.0)**2).mean
        end
      end.new
    end

    it "executes the shielded action and records the shield's metrics per episode" do
      TorchTestHelpers.with_torch_seed do
        environment = ContinuousTarget.new(low: 0.0, high: 10.0)
        agent = subject.new(environment, shield: shield, initial_log_std: 1.0, ppo_epochs: 2, minibatch_size: 8)
        agent.train(episodes: 3, steps_per_episode: 8)

        # the environment receives the shield's action, scaled onto the box
        expect(environment.actions_taken.uniq).to be == [[7.5]]
        # the shield receives observations and base actions in [-1, 1]
        expect(shield.calls.size).to be == 24
        expect(shield.calls.all? { |state, action| state.size == 1 && action.size == 1 && action[0].abs <= 1.0 }).to be == true
        expect(shield.calls.map { |_, action| action[0] }.uniq.size).to be > 1
        # one-step episodes: one intervention per episode
        expect(agent.metrics[:shield_interventions]).to be == Array.new(24, 1.0)
        expect(agent.metrics[:safety_loss].size).to be == 3 * 2
        expect(shield.loss_calls).to be == 6
        expect(agent.predict([0.3])).to be == [7.5]
        # without the shield, the policy's own action
        unshielded = agent.predict([0.3], shielded: false)
        expect(unshielded).not.to be == [7.5]
        expect(unshielded[0].between?(0.0, 10.0)).to be == true
      end
    end

    it "minimizes the safety loss with the policy's loss" do
      TorchTestHelpers.with_torch_seed do
        environment = ContinuousTarget.new
        # rewards don't depend on the base action (the shield fixes it), so
        # only the safety loss moves the policy's mean
        agent = subject.new(environment, shield: shield, safety_weight: 1.0, learning_rate: 1e-2, ppo_epochs: 4, minibatch_size: 16)
        before = Torch.no_grad { Torch.tanh(agent.agent.policy_model.call(Torch.tensor([[0.0]]))).item }
        agent.train(episodes: 40, steps_per_episode: 16)
        after = Torch.no_grad { Torch.tanh(agent.agent.policy_model.call(Torch.tensor([[0.0]]))).item }

        expect(after).to be < before - 0.5
      end
    end

    it "scores sampled actions, or the deterministic ones without samples" do
      TorchTestHelpers.with_torch_seed do
        subject.new(ContinuousTarget.new, shield: shield, safety_samples: 4, ppo_epochs: 1, minibatch_size: 8).train(episodes: 1, steps_per_episode: 8)
        subject.new(ContinuousTarget.new, shield: shield, safety_samples: 0, ppo_epochs: 1, minibatch_size: 8).train(episodes: 1, steps_per_episode: 8)

        expect(shield.shapes).to be == [[4, 8, 1], [8, 1]]
      end
    end

    it "reaches the standard deviation with sampled actions" do
      gradient = lambda do |samples|
        TorchTestHelpers.with_torch_seed do
          agent = subject.new(ContinuousTarget.new, shield: shield, safety_samples: samples)
          shield.safety_loss(Torch.tensor([[0.2], [-0.5]]), agent.safety_actions(Torch.tensor([[0.2], [-0.5]]))).backward
          [agent.agent.log_std.grad&.abs&.sum&.item.to_f, agent.agent.policy_model.parameters.sum { |p| p.grad.abs.sum.item }]
        end
      end

      sampled_std, sampled_mean = gradient.call(8)
      deterministic_std, deterministic_mean = gradient.call(0)
      expect(sampled_std).to be > 0.0
      expect(sampled_mean).to be > 0.0
      # the deterministic action doesn't depend on the spread
      expect(deterministic_std).to be == 0.0
      expect(deterministic_mean).to be > 0.0
    end

    it "is only for continuous actions" do
      discrete = Class.new do
        include Reinforce::Environment

        def state_size = 1

        def actions = [0, 1]
      end.new

      expect { subject.new(discrete, shield: shield) }.to raise_exception(ArgumentError, message: be =~ /DiscreteShield/)
    end
  end

  with "a target KL divergence" do
    def updates(target_kl)
      TorchTestHelpers.with_torch_seed do
        agent = subject.new(ContinuousTarget.new, target_kl: target_kl, learning_rate: 5e-2, ppo_epochs: 4, minibatch_size: 8)
        agent.train(episodes: 3, steps_per_episode: 32)
        agent.metrics[:approx_kl].size
      end
    end

    it "stops a rollout's updates once the policy moves too far" do
      # 4 epochs of 4 minibatches per rollout, unless stopped
      expect(updates(nil)).to be == 3 * 4 * 4
      expect(updates(1e-6)).to be < 3 * 4 * 4
    end

    it "records the gradient steps of each rollout and whether it stopped early" do
      metrics = lambda do |target_kl|
        TorchTestHelpers.with_torch_seed do
          agent = subject.new(ContinuousTarget.new, target_kl: target_kl, learning_rate: 5e-2, ppo_epochs: 4, minibatch_size: 8)
          agent.train(episodes: 3, steps_per_episode: 32)
          [agent.metrics[:updates], agent.metrics[:early_stop], agent.metrics[:approx_kl].size]
        end
      end

      updates, stops, = metrics.call(nil)
      expect(updates).to be == [16, 16, 16]
      expect(stops).to be == [0, 0, 0]
      updates, stops, measured = metrics.call(1e-6)
      expect(updates.all? { |count| count < 16 }).to be == true
      expect(stops).to be == [1, 1, 1]
      # every minibatch measured is updated, except the one that stops each rollout
      expect(updates.sum).to be == measured - 3
    end
  end

  with "episodes that end at a time limit" do
    # The reward is always 1 and every step ends the episode, at a time
    # limit (truncated) or not: the value of a state is then 1 / (1 - γ)
    # with bootstrapping, and 1 without.
    def environment(truncated)
      Class.new do
        include Reinforce::Environment

        define_method(:truncated) { truncated }

        def state_size = 1

        def action_space = Reinforce::Spaces::Box.new(low: -1.0, high: 1.0, size: 1)

        def reset = [0.0]

        def step(_action) = [[0.0], 1.0, true, {truncated: truncated, metrics: {requests: 2}}]
      end.new
    end

    def value_after_training(truncated)
      TorchTestHelpers.with_torch_seed do
        agent = subject.new(environment(truncated), normalize_rewards: false, discount_factor: 0.5, learning_rate: 1e-2,
          ppo_epochs: 4, minibatch_size: 16)
        agent.train(episodes: 60, steps_per_episode: 16)
        [Torch.no_grad { agent.agent.get_value(Torch.tensor([[0.0]])).item }, agent]
      end
    end

    it "bootstraps from the state reached instead of treating it as terminal" do
      terminal, = value_after_training(false)
      bootstrapped, agent = value_after_training(true)

      expect(terminal).to be_within(0.1).of(1.0)
      expect(bootstrapped).to be_within(0.2).of(2.0)
      # the environment's metrics are summed per episode
      expect(agent.metrics[:episode_requests].uniq).to be == [2.0]
    end
  end
end
