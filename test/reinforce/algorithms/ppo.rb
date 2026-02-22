# frozen_string_literal: true

require "reinforce/algorithms/ppo"

describe Reinforce::Algorithms::PPO do
  let(:environment) do
    env = Object.new
    env.define_singleton_method(:state_size) { 2 }
    env.define_singleton_method(:actions) { [:left, :right] }
    env
  end

  let(:agent) { Reinforce::Algorithms::PPO.new(environment, 0.001, nil, nil, 0.2, 1, 2, 1.0) }

  it "builds rollout buffers with step-aligned shapes" do
    buffers = agent.build_rollout_buffers(4)

    expect(buffers[:obs].size.to_a).to be == [4, 2]
    expect(buffers[:actions].size.to_a).to be == [4]
    expect(buffers[:logprobs].size.to_a).to be == [4]
    expect(buffers[:rewards].size.to_a).to be == [4]
    expect(buffers[:dones].size.to_a).to be == [4]
    expect(buffers[:values].size.to_a).to be == [4]
  end

  it "computes gae correctly when rollout ends on terminal transition" do
    agent.instance_variable_set(:@gaelam, 1.0)

    rewards = Torch.tensor([1.0, 1.0], dtype: :float32)
    values = Torch.tensor([0.5, 0.5], dtype: :float32)
    dones = Torch.tensor([0.0, 0.0], dtype: :float32)

    advantages, returns = agent.compute_gae(rewards, values, dones, 42.0, true)

    expect(advantages.to_a).to be == [1.5, 0.5]
    expect(returns.to_a).to be == [2.0, 1.0]
  end

  it "cuts off advantage propagation across mid-rollout terminal boundaries" do
    agent.instance_variable_set(:@gaelam, 1.0)

    rewards = Torch.tensor([1.0, 1.0, 1.0], dtype: :float32)
    values = Torch.tensor([0.0, 0.0, 0.0], dtype: :float32)
    dones = Torch.tensor([0.0, 1.0, 0.0], dtype: :float32)

    advantages, returns = agent.compute_gae(rewards, values, dones, 2.0, false)

    expect(advantages.to_a).to be == [1.0, 4.0, 3.0]
    expect(returns.to_a).to be == [1.0, 4.0, 3.0]
  end
end
