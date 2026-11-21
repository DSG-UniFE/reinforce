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
end
