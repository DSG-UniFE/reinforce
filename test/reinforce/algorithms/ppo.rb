# frozen_string_literal: true

require "reinforce/algorithms/ppo"
require_relative "../../support/agent_contract"
require_relative "../../support/torch_test_helpers"

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

  it "splits rollout indices into non-overlapping minibatches of minibatch_size" do
    # Regression: minibatches were sliced with an inclusive range
    # (b_inds[start..start + minibatch_size]), so each had minibatch_size + 1
    # samples and consecutive minibatches overlapped by one.
    indices = [4, 0, 3, 1, 2]
    batches = agent.minibatches(indices)

    expect(batches).to be == [[4, 0], [3, 1], [2]]
    expect(batches.flatten).to be == indices
  end

  it "lets the entropy bonus affect training" do
    # Regression: #train re-wrapped the entropy tensor in Torch.tensor(...),
    # which drops its autograd history, so the entropy term contributed no
    # gradient: training with and without it produced identical weights.
    steppable = Object.new
    steppable.define_singleton_method(:state_size) { 2 }
    steppable.define_singleton_method(:actions) { %i[left right] }
    steppable.define_singleton_method(:reset) { [0.1, 0.2] }
    steppable.define_singleton_method(:step) { |action| [[0.1, 0.2], (action == 1) ? 1.0 : 0.0, false, {}] }

    policy_weights = [0.0, 0.5].map do |coefficient|
      TorchTestHelpers.with_torch_seed do
        ppo = Reinforce::Algorithms::PPO.new(steppable, 0.01, nil, nil, 0.2, 1, 4, 0.99, entropy_coefficient: coefficient)
        ppo.train(episodes: 2, steps_per_episode: 8)
        TorchTestHelpers.parameter_snapshot(ppo.agent.policy_model).flatten
      end
    end

    expect(policy_weights[0].zip(policy_weights[1]).any? { |a, b| (a - b).abs > 1e-6 }).to be == true
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
