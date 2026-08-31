# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi.

require 'reinforce/q_function_ann'

describe Reinforce::QFunctionANN do
  let(:q_function) { Reinforce::QFunctionANN.new(2, 2, 0.01, 0.99) }
  let(:q_target_function) { Reinforce::QFunctionANN.new(2, 2, 0.01, 0.99) }

  it 'can be soft updated from another ANN-based Q function' do
    # Regression test for two pre-existing bugs found while extracting
    # Reinforce::Networks.soft_update!:
    #   1. `p.flatten.to_a do |value| ... end` never actually ran the
    #      block below it (Tensor#to_a does not yield), so this test
    #      passed regardless of whether soft_update was correct.
    #   2. The expected value was computed as
    #      target_old * tau + online * (1 - tau), which is the reverse of
    #      what #soft_update actually (and correctly) computes:
    #      target_old * (1 - tau) + online * tau -- see
    #      Reinforce::Networks.soft_update! for the convention this
    #      follows (tau is how much of `online` is blended in).
    tau = 0.9
    q_copied_params = q_function.parameters.map(&:clone)
    qtar_copied_params = q_target_function.parameters.map(&:clone)
    q_target_function.soft_update(q_function, tau)
    q_target_function.parameters.each_with_index do |p, i|
      expected = (qtar_copied_params[i] * (1.0 - tau)) + (q_copied_params[i] * tau)
      p.flatten.to_a.zip(expected.flatten.to_a).each do |value, expected_value|
        expect(value).to be_within(1E-3).of(expected_value)
      end
    end
  end

  describe '#update' do
    it 'bootstraps from next_action (not the greedy action) when on_policy: true' do
      # Regression test for the SARSA-vs-Q-learning bug found while
      # reconciling TemporalDifference and SARSA: #update used to always
      # bootstrap from the greedy action under the current Q-network
      # (next_q_values.argmax), even when called from SARSA's on-policy
      # training loop, which had already computed the actual next_action
      # taken by the behavior policy and thrown it away. See
      # lib/reinforce/algorithms/sarsa.rb and
      # lib/reinforce/algorithms/temporal_difference.rb.
      Torch.manual_seed(42)
      q_on_policy = Reinforce::QFunctionANN.new(2, 2, 0.01, 0.5)
      q_off_policy = Reinforce::QFunctionANN.new(2, 2, 0.01, 0.5)
      # Give both networks identical weights, so the only difference
      # between the two #update calls below is the on_policy: flag.
      q_off_policy.load_state_dict(q_on_policy.state_dict)

      next_state = [[0.3, -0.7]]
      next_q_values = Torch.no_grad { q_on_policy.forward(next_state) }
      greedy_next_action = next_q_values.argmax(1).to_a.first
      # With only two actions, "the other one" is guaranteed not to be the
      # greedy action -- exactly the scenario the old code silently ignored.
      other_next_action = 1 - greedy_next_action

      # Sanity check: the two candidate bootstrap values must actually
      # differ, or this test couldn't distinguish the two code paths.
      greedy_value = next_q_values[0][greedy_next_action].item
      other_value = next_q_values[0][other_next_action].item
      expect(greedy_value == other_value).to be == false

      experience = {
        state: [[0.1, 0.2]],
        action: [0],
        next_state: next_state,
        next_action: [other_next_action],
        reward: [0.0],
        done: [false]
      }

      loss_on_policy = q_on_policy.update(experience, on_policy: true)
      loss_off_policy = q_off_policy.update(experience, on_policy: false)

      expect(loss_on_policy == loss_off_policy).to be == false
    end
  end
end
