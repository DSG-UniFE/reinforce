# frozen_string_literal: true

require "reinforce/models/dueling_q_network"
require "reinforce/networks"

describe Reinforce::Models::DuelingQNetwork do
  let(:small_config) { Reinforce::Networks::Config.new(hidden_size: 8) }

  it "produces one Q-value per action with the expected batch shape" do
    network = Reinforce::Models::DuelingQNetwork.new(3, 4, network_config: small_config)
    states = Torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype: :float32)

    q_values = network.forward(states)

    expect(q_values.size.to_a).to be == [2, 4]
  end

  it "combines value and advantage streams so the advantage term is mean-zero per state" do
    # Q(s, a) = V(s) + (A(s, a) - mean_a A(s, a)) -- subtracting the mean
    # advantage is what keeps V and A identifiable (see the class comment
    # for why). A direct consequence: averaging Q(s, ·) across actions for
    # a given state must recover V(s) exactly, since the advantage terms
    # cancel out by construction.
    network = Reinforce::Models::DuelingQNetwork.new(2, 3, network_config: small_config)
    state = Torch.tensor([[0.5, -0.5]], dtype: :float32)

    q_values = Torch.no_grad { network.forward(state) }
    mean_q = q_values.mean.item

    value = Torch.no_grad { network.instance_variable_get(:@value_head).call(network.instance_variable_get(:@shared).call(state)) }
    expect(mean_q).to be_within(1e-4).of(value.item)
  end

  it "handles an unbatched single state, not just a [batch, features] tensor" do
    # Regression test: the advantage-mean subtraction originally hardcoded
    # dim: 1, which assumes a batch dimension always exists. That broke
    # DQN#choose_action/#predict, which call #forward with a single,
    # unbatched state ([num_actions], no leading batch dim) rather than a
    # minibatch -- raising "Dimension out of range" the first time a
    # DuelingQNetwork-backed DQN tried to act, not just when training.
    network = Reinforce::Models::DuelingQNetwork.new(3, 4, network_config: small_config)

    q_values = network.forward(Torch.tensor([0.0, 1.0, 0.0], dtype: :float32))

    expect(q_values.size.to_a).to be == [4]
  end

  it "reshapes its trunk from network_config: instead of a fixed depth/width" do
    default_network = Reinforce::Models::DuelingQNetwork.new(3, 4)
    deep_network = Reinforce::Models::DuelingQNetwork.new(
      3, 4, network_config: Reinforce::Networks::Config.new(hidden_size: [16, 8, 4])
    )

    # Default: 2 hidden layers (Reinforce::Networks::Config's default) in
    # @shared, plus the value and advantage heads -- 2*2 + 2*2 = 8
    # parameter tensors (weight + bias per Linear).
    expect(default_network.parameters.size).to be == 8
    # Deep: 3 hidden layers instead of 2, so 2 more Linear layers than
    # default -- 3*2 + 2*2 = 10.
    expect(deep_network.parameters.size).to be == 10
  end
end
