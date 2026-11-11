# frozen_string_literal: true

require "reinforce"
require_relative "../../support/agent_contract"
require_relative "../../support/torch_test_helpers"

describe Reinforce::Algorithms::SuccessorFeatures do
  # A 1x5 corridor with goals at both ends; features [left, right, step].
  def corridor(**options)
    blocked = (1..2).flat_map { |row| (0..4).map { |column| [row, column] } }
    Reinforce::Environments::MultiGoalGridWorld.new(5, [0, 1], [[0, 0], [0, 4]], blocked, observation: :one_hot, **options)
  end

  def basis
    [[1.0, 0.0, -0.01], [0.0, 1.0, -0.01]]
  end

  include_context AgentContract,
    factory: -> {
      blocked = (1..2).flat_map { |row| (0..4).map { |column| [row, column] } }
      environment = Reinforce::Environments::MultiGoalGridWorld.new(5, [0, 1], [[0, 0], [0, 4]], blocked)
      Reinforce::Algorithms::SuccessorFeatures.new(environment, tasks: [[1.0, 0.0, 0.0]])
    },
    state: [0, 1],
    actions: %i[up down left right]

  it "computes GPI values as the max over policies of psi(s, a) . task" do
    TorchTestHelpers.with_torch_seed do
      agent = Reinforce::Algorithms::SuccessorFeatures.new(corridor, tasks: basis)
      state = corridor.reset
      task = [0.3, -0.7, 0.2]

      per_policy = agent.psi_networks.map do |network|
        psi = Torch.no_grad { network.call(Torch.tensor([state], dtype: :float32)) }.reshape(4, 3)
        psi.matmul(Torch.tensor(task)).to_a
      end
      expected = per_policy.transpose.map(&:max)

      expect(agent.q_values(state, task:).zip(expected).all? { |actual, wanted| (actual - wanted).abs < 1e-5 }).to be == true
      expect(agent.q_values(state, task:, policies: [1]).zip(per_policy[1]).all? { |actual, wanted| (actual - wanted).abs < 1e-5 }).to be == true
    end
  end

  it "learns from features alone and transfers zero-shot to new preferences" do
    TorchTestHelpers.with_torch_seed do
      agent = Reinforce::Algorithms::SuccessorFeatures.new(corridor(random_start: true), tasks: basis, warmup_steps: 100)

      agent.train(episodes: 300, steps_per_episode: 10)
      start = corridor.reset

      expect(agent.predict(start, task: [1.0, 0.0, -0.01])).to be == :left
      expect(agent.predict(start, task: [0.0, 1.0, -0.01])).to be == :right
      # Unseen: the near goal is now penalized, the far one still worth it.
      expect(agent.predict(start, task: [-1.0, 0.5, -0.01])).to be == :right
      expect(agent.metrics[:loss]).not.to be(:empty?)
    end
  end

  it "requires the environment to report transition features" do
    environment = Reinforce::Environments::GridWorld.new(3, [0, 0], [2, 2], [])
    agent = Reinforce::Algorithms::SuccessorFeatures.new(environment, tasks: [[1.0]])

    expect { agent.train(episodes: 1, steps_per_episode: 1) }.to raise_exception(ArgumentError, message: be =~ /features/)
  end

  it "needs at least one training task" do
    expect { Reinforce::Algorithms::SuccessorFeatures.new(corridor, tasks: []) }.to raise_exception(ArgumentError)
  end
end
