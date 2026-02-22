# frozen_string_literal: true

require "reinforce/algorithms/dqn"

describe Reinforce::Algorithms::DQN do
  let(:environment) do
    env = Object.new
    env.define_singleton_method(:state_size) { 2 }
    env.define_singleton_method(:actions) { [:left, :right] }
    env
  end

  let(:agent) { Reinforce::Algorithms::DQN.new(environment, 0.001, 1.0) }

  it "computes td targets with terminal masking" do
    next_q_values = Torch.tensor([[2.0, 10.0], [3.0, 4.0]], dtype: :float32)
    targets = agent.compute_td_targets(next_q_values, [1.0, 2.0], [true, false]).to_a

    expect(targets[0]).to be == 1.0
    expect(targets[1]).to be == 6.0
  end

  it "uses per-row max q-values, not a global max" do
    next_q_values = Torch.tensor([[2.0, 10.0], [3.0, 4.0]], dtype: :float32)
    targets = agent.compute_td_targets(next_q_values, [0.0, 0.0], [false, false]).to_a

    expect(targets).to be == [10.0, 4.0]
  end

  it "selects q-values using action indices robustly" do
    q_values = Torch.tensor([[1.0, 5.0], [7.0, 2.0]], dtype: :float32)
    actions = Torch.tensor([1.0, 0.0], dtype: :float32)

    selected = agent.q_values_for_actions(q_values, actions).to_a
    expect(selected).to be == [5.0, 7.0]
  end
end
