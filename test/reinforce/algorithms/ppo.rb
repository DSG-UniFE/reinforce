# frozen_string_literal: true

require "reinforce/algorithms/ppo"
require_relative "../../support/agent_contract"

describe Reinforce::Algorithms::PPO do
  let(:environment) do
    env = Object.new
    env.define_singleton_method(:state_size) { 2 }
    env.define_singleton_method(:actions) { [:left, :right] }
    env
  end

  let(:agent) { Reinforce::Algorithms::PPO.new(environment, 0.001, nil, nil, 0.2, 1, 2, 1.0) }

  include_context AgentContract, factory: -> { agent }, state: [0.1, 0.2], actions: %i[left right]

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

  it "predicts the same action from an Array state and the equivalent Tensor state" do
    # Regression: #predict converted its input with
    # `argument = Torch.tensor(...) unless state.is_a?(Torch::Tensor)`, which
    # left argument nil -- and crashed the policy network -- whenever the
    # state was already a Tensor. #predict is greedy by default, so equal
    # actions also check both input forms reach the policy as the same values.
    array_action = agent.predict([0.1, 0.2])
    tensor_action = agent.predict(Torch.tensor([0.1, 0.2], dtype: :float32))

    expect(environment.actions).to be(:include?, array_action)
    expect(tensor_action).to be == array_action
  end
end
