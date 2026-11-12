# frozen_string_literal: true

require "reinforce/shields"
require_relative "../../support/torch_test_helpers"

describe "ContinuousSAC with a multi-cluster shield" do
  it "trains with shielded actions and the safety loss" do
    TorchTestHelpers.with_torch_seed do
      environment = Reinforce::Environments::MultiCluster.hard(clusters: 3, horizon: 40)
      safety = Reinforce::Shields::MultiClusterSafety.new(environment)
      shield = Reinforce::Shields::AllocationShield.new(safety, preference: :latency)
      agent = Reinforce::Algorithms::ContinuousSAC.new(environment, shield: shield, warmup_steps: 40, batch_size: 32, hidden_size: 32)
      agent.train(episodes: 2, steps_per_episode: 40)
      metrics = agent.metrics

      expect(metrics[:safety_loss].size).to be == 41
      expect(metrics[:safety_loss].all?(&:finite?)).to be == true
      expect(metrics[:shield_interventions].sum).to be >= metrics[:shield_corrections].sum
      expect(metrics[:shield_safety].first).to be > 0.0
      expect(metrics[:episode_requested].size).to be == 2
      expect(environment.action_space.contains?(agent.predict(environment.reset))).to be == true
    end
  end
end
