# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi.

require "reinforce/q_function_ann"

describe Reinforce::QFunctionANN do
  let(:q_function) { Reinforce::QFunctionANN.new(2, 2, 0.01, 0.99) }
  let(:q_target_function) { Reinforce::QFunctionANN.new(2, 2, 0.01, 0.99) }

  it "can be soft updated from another ANN-based Q function" do
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

  describe "#compute_td_targets" do
    it "bootstraps reward + discount * Q(next_state, next_action), or just reward when done" do
      next_q_values = Torch.tensor([[2.0, 10.0], [3.0, 4.0]], dtype: :float32)

      targets = q_function.compute_td_targets(next_q_values, [1, 0], [1.0, 2.0], [true, false]).to_a

      # Row 0 is done, so its target is just the reward (1.0), ignoring
      # next_q_values entirely. Row 1 bootstraps from action 0's value:
      # 2.0 + 0.99 * 3.0 = 4.97.
      expect(targets[0]).to be_within(1e-4).of(1.0)
      expect(targets[1]).to be_within(1e-4).of(4.97)
    end

    it "bootstraps from the given next_actions, not always the greedy one" do
      next_q_values = Torch.tensor([[2.0, 10.0]], dtype: :float32)

      greedy_target = q_function.compute_td_targets(next_q_values, [1], [0.0], [false]).to_a.first
      other_target = q_function.compute_td_targets(next_q_values, [0], [0.0], [false]).to_a.first

      expect(greedy_target == other_target).to be == false
    end
  end

  describe "#q_values_for_actions" do
    it "selects q-values using action indices robustly" do
      q_values = Torch.tensor([[1.0, 5.0], [7.0, 2.0]], dtype: :float32)
      actions = Torch.tensor([1.0, 0.0], dtype: :float32)

      selected = q_function.q_values_for_actions(q_values, actions).to_a
      expect(selected).to be == [5.0, 7.0]
    end
  end

  describe "#update" do
    it "bootstraps from next_action (not the greedy action) when on_policy: true" do
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

      loss_on_policy = q_on_policy.update(experience, on_policy: true)[:loss]
      loss_off_policy = q_off_policy.update(experience, on_policy: false)[:loss]

      expect(loss_on_policy == loss_off_policy).to be == false
    end

    it "bootstraps from the target network's own greedy value when target: is given" do
      online_layer = Torch::NN::Linear.new(2, 2)
      Torch::NN::Init.constant!(online_layer.weight, 0.0)
      Torch::NN::Init.constant!(online_layer.bias, 0.0)
      q = Reinforce::QFunctionANN.new(2, 2, 0.01, 0.5, architecture: Torch::NN::Sequential.new(online_layer))

      target_double = Object.new
      target_double.define_singleton_method(:forward) { |_next_state| Torch.tensor([[3.0, 9.0]]) }

      experience = {state: [[0.0, 0.0]], action: [0], next_state: [[0.0, 0.0]], reward: [0.0], done: [false]}

      result = q.update(experience, target: target_double)

      # predicted Q(state, action 0) is 0.0 (weight/bias both zeroed), so
      # td_error == the target itself: reward(0) + 0.5 * target's own
      # greedy value (action 1, 9.0) = 4.5.
      expect(result[:td_errors].first).to be_within(1e-4).of(4.5)
    end

    it "double_dqn: true evaluates the online-selected action using target, not target's own greedy value" do
      online_layer = Torch::NN::Linear.new(2, 2)
      Torch::NN::Init.constant!(online_layer.weight, 0.0)
      # Weight is zero, so this bias is Q(any_state, ·): online favors
      # action 0 regardless of input.
      Torch.no_grad { online_layer.bias.copy!(Torch.tensor([1.0, 0.0])) }
      q = Reinforce::QFunctionANN.new(2, 2, 0.01, 0.5, architecture: Torch::NN::Sequential.new(online_layer))

      target_double = Object.new
      # target's own greedy action is 1 (100.0 > 5.0) -- naively trusting
      # it would bootstrap from 100.0. Double DQN must instead evaluate
      # *online's* chosen action (0) under target, i.e. 5.0.
      target_double.define_singleton_method(:forward) { |_next_state| Torch.tensor([[5.0, 100.0]]) }

      experience = {state: [[0.0, 0.0]], action: [0], next_state: [[0.0, 0.0]], reward: [0.0], done: [false]}

      result = q.update(experience, target: target_double, double_dqn: true)

      # predicted Q(state, action 0) = 1.0 (the bias); target = 0 + 0.5 * 5.0 = 2.5.
      expect(result[:td_errors].first).to be_within(1e-4).of(1.5)
      expect(result[:loss]).to be_within(1e-4).of(1.5 * 1.5)
    end

    it "requires target: when double_dqn: true" do
      experience = {state: [[0.0, 0.0]], action: [0], next_state: [[0.0, 0.0]], reward: [0.0], done: [false]}

      expect { q_function.update(experience, double_dqn: true) }.to raise_exception(ArgumentError)
    end

    it "weights the loss by weights: instead of treating every sample equally" do
      Torch.manual_seed(1)
      q_unweighted = Reinforce::QFunctionANN.new(2, 2, 0.01, 0.5)
      q_weighted = Reinforce::QFunctionANN.new(2, 2, 0.01, 0.5)
      q_weighted.load_state_dict(q_unweighted.state_dict)

      experience = {
        state: [[0.1, 0.2], [0.9, -0.4]],
        action: [0, 1],
        next_state: [[0.3, -0.1], [0.2, 0.5]],
        reward: [1.0, -1.0],
        done: [false, false]
      }

      loss_unweighted = q_unweighted.update(experience.dup, weights: [1.0, 1.0])[:loss]
      loss_weighted = q_weighted.update(experience.dup, weights: [1.0, 0.0])[:loss]

      # Zeroing out the second sample's weight must change the loss versus
      # weighting both samples equally -- otherwise weights: isn't doing
      # anything.
      expect(loss_unweighted == loss_weighted).to be == false
    end
  end
end
