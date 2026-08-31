# frozen_string_literal: true

require_relative "../../support/torch_test_helpers"
require "reinforce"

describe "SAC real-environment smoke test" do
  def build_network(input_size, output_size)
    Torch::NN::Sequential.new(
      Torch::NN::Linear.new(input_size, 16),
      Torch::NN::ReLU.new,
      Torch::NN::Linear.new(16, output_size)
    )
  end

  it "trains on GridWorld and records finite metrics" do
    TorchTestHelpers.with_torch_seed do
      environment = Reinforce::Environments::GridWorld.new(3, [0, 0], [0, 1], 0)
      agent = Reinforce::Algorithms::SAC.new(
        environment,
        policy_model: build_network(2, 4),
        q1_model: build_network(2, 4),
        q2_model: build_network(2, 4),
        q1_target_model: build_network(2, 4),
        q2_target_model: build_network(2, 4),
        learning_rate: 0.01,
        batch_size: 8,
        replay_size: 32,
        warmup_steps: 8
      )

      agent.train(episodes: 12, steps_per_episode: 3)

      expect(agent.logs[:episode_reward].size).to be == 12
      expect(agent.logs[:loss].empty?).to be == false
      expect(agent.logs[:loss].all?(&:finite?)).to be == true
      expect(agent.logs[:entropy].all?(&:finite?)).to be == true
    end
  end
end
