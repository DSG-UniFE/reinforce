# frozen_string_literal: true

require "reinforce"
require_relative "../../support/agent_contract"
require_relative "../../support/torch_test_helpers"

describe Reinforce::Algorithms::DIAYN do
  def grid
    Reinforce::Environments::GridWorld.new(3, [0, 0], [0, 1], [], reward: :sparse)
  end

  def build(**options)
    Reinforce::Algorithms::DIAYN.new(grid, skills: 4, **options)
  end

  include_context AgentContract,
    factory: -> { Reinforce::Algorithms::DIAYN.new(Reinforce::Environments::GridWorld.new(3, [0, 0], [2, 2], []), skills: 2) },
    state: [0, 0],
    actions: %i[up down left right]

  it "appends a one-hot skill to observations, rejecting unknown skills" do
    diayn = build

    expect(diayn.with_skill([1, 2], 2)).to be == [1.0, 2.0, 0.0, 0.0, 1.0, 0.0]
    expect { diayn.with_skill([1, 2], 4) }.to raise_exception(ArgumentError)
  end

  it "rewards reaching a state by log q(skill | state) - log p(skill)" do
    TorchTestHelpers.with_torch_seed do
      diayn = build
      probabilities = diayn.skill_probabilities([1, 1])

      expect(diayn.skill_reward([1, 1], 3)).to be_within(1e-5).of(Math.log(probabilities[3]) + Math.log(4))
      expect(probabilities.sum).to be_within(1e-5).of(1.0)
    end
  end

  describe "its skill environment" do
    it "samples a skill per episode and replaces the reward with DIAYN's, keeping the task's in the info" do
      TorchTestHelpers.with_torch_seed do
        diayn = build
        environment = diayn.skill_environment

        observation = environment.reset
        skill = environment.skill
        next_observation, reward, done, info = environment.step(:right) # reaches the task's goal

        expect(environment.state_size).to be == 2 + 4
        expect(observation).to be == diayn.with_skill([0, 0], skill)
        expect(next_observation).to be == diayn.with_skill([0, 1], skill)
        expect(reward).to be_within(1e-5).of(diayn.skill_reward([0, 1], skill))
        expect(info).to be == {extrinsic_reward: 1, skill:}
        expect(done).to be == false # ignore_done: a skill runs to the step limit
      end
    end

    it "passes the task's termination through when ignore_done is off" do
      environment = build(ignore_done: false).skill_environment
      environment.reset

      expect(environment.step(:right)[2]).to be == true
    end

    it "trains the discriminator on the finished episode's states at the next reset" do
      TorchTestHelpers.with_torch_seed do
        diayn = build(discriminator_updates: 3)
        environment = diayn.skill_environment
        snapshot = TorchTestHelpers.parameter_snapshot(diayn.discriminator)

        environment.reset
        2.times { environment.step(:down) }
        expect(diayn.metrics[:discriminator_loss]).to be == []

        environment.reset

        expect(diayn.metrics[:discriminator_loss].size).to be == 3
        expect(diayn.metrics[:discriminator_accuracy].all? { |accuracy| accuracy.between?(0.0, 1.0) }).to be == true
        expect(TorchTestHelpers.parameters_changed?(diayn.discriminator, snapshot)).to be == true
      end
    end
  end

  it "learns skills the discriminator can tell apart" do
    TorchTestHelpers.with_torch_seed do
      diayn = Reinforce::Algorithms::DIAYN.new(
        Reinforce::Environments::GridWorld.new(5, [2, 2], [4, 4], []), skills: 2,
        agent: ->(environment) { Reinforce::Algorithms::PPO.new(environment, learning_rate: 1e-2) }
      )

      diayn.train(episodes: 200, steps_per_episode: 10)

      # Chance is 0.5; seeds 1, 2 and 12345 reach 0.84-0.90 (0.59-0.68 at
      # half the budget).
      accuracy = diayn.metrics[:discriminator_accuracy]
      expect(accuracy.last(20).sum / 20).to be > 0.75
      expect(diayn.metrics[:skill_reward].size).to be == 2000
    end
  end

  describe "discriminator features" do
    # Only the row matters: the discriminator sees [row].
    def by_row
      Reinforce::Algorithms::DIAYN.new(grid, skills: 4, discriminator_features: ->(state) { [state[0]] })
    end

    it "builds the discriminator over the features, not the full state" do
      expect(by_row.discriminator.parameters.first.shape).to be == [64, 1]
    end

    it "judges states only by their features" do
      TorchTestHelpers.with_torch_seed do
        diayn = by_row

        expect(diayn.skill_probabilities([1, 0])).to be == diayn.skill_probabilities([1, 2])
        expect(diayn.skill_reward([1, 0], 2)).to be_within(1e-6).of(diayn.skill_reward([1, 2], 2))
      end
    end

    it "trains the discriminator on features, leaving the policy's observations unchanged" do
      TorchTestHelpers.with_torch_seed do
        diayn = by_row
        environment = diayn.skill_environment
        observation = environment.reset

        environment.step(:down)
        environment.reset

        expect(observation.size).to be == 2 + 4
        expect(diayn.metrics[:discriminator_loss].size).to be == diayn.config.discriminator_updates
      end
    end
  end
end
