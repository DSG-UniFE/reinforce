# frozen_string_literal: true

require "reinforce/shields"
require_relative "../../support/torch_test_helpers"

describe "PPO with a multi-cluster shield" do
  it "trains with shielded actions and the safety loss" do
    TorchTestHelpers.with_torch_seed do
      environment = Reinforce::Environments::MultiCluster.hard(clusters: 3, horizon: 40)
      safety = Reinforce::Shields::MultiClusterSafety.new(environment)
      shield = Reinforce::Shields::AllocationShield.new(safety, preference: :latency)
      agent = Reinforce::Algorithms::PPO.new(environment, shield: shield, ppo_epochs: 2, minibatch_size: 32)
      # rollouts of 60 steps: episodes end at the horizon (40) and at the
      # end of each rollout
      agent.train(episodes: 2, steps_per_episode: 60)
      metrics = agent.metrics

      expect(metrics[:safety_loss].size).to be == 2 * 2 * 2
      expect(metrics[:safety_loss].all?(&:finite?)).to be == true
      expect(metrics[:episode_requested].size).to be == 4
      expect(metrics[:shield_interventions].sum).to be >= metrics[:shield_corrections].sum
      expect(metrics[:shield_safety].first).to be > 0.0
      expect(environment.action_space.contains?(agent.predict(environment.reset))).to be == true
    end
  end
end
