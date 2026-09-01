# frozen_string_literal: true

require "reinforce/models/discrete_policy_network"

describe Reinforce::Models::DiscretePolicyNetwork do
  it "produces logits/probabilities with expected shapes and valid actions" do
    policy = Reinforce::Models::DiscretePolicyNetwork.new(2, %i[left right], hidden_size: 16)
    states = Torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype: :float32)

    logits = policy.forward(states)
    probs = policy.probabilities(states)
    action = policy.predict([0.0, 1.0])

    expect(logits.size.to_a).to be == [2, 2]
    expect(probs.size.to_a).to be == [2, 2]
    expect(%i[left right].include?(action)).to be == true
  end
end
