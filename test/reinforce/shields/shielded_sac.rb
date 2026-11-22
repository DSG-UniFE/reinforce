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
      state = environment.reset
      expect(environment.action_space.contains?(agent.predict(state))).to be == true
      # without the shield: the policy's own action, and the shield isn't called
      calls = metrics[:shield_safety].size
      shield.define_singleton_method(:call) { |*| raise "the shield was called" }
      expect(environment.action_space.contains?(agent.predict(state, shielded: false))).to be == true
      expect(metrics[:shield_safety].size).to be == calls
    end
  end
end
