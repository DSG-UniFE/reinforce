# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi.

require "reinforce/algorithms/monte_carlo_policy_gradient"
require_relative "../../support/torch_test_helpers"

describe Reinforce::Algorithms::MonteCarloPolicyGradient do
  let(:num_states) { 4 }
  let(:num_actions) { 2 }
  let(:discount_factor) { 0.99 }
  let(:model) { Object.new }
  let(:optimizer) { Object.new }
  let(:agent) { Reinforce::Algorithms::MonteCarloPolicyGradient.new(num_states, num_actions, discount_factor, model, optimizer) }

  it "can be instantiated" do
    expect(agent).not.to be_nil
  end

  it "can be saved" do
    expect(agent.respond_to?(:save)).to be == true
  end

  it "can be loaded" do
    expect(agent.respond_to?(:load)).to be == true
  end

  describe "with a real policy network" do
    let(:policy) { Torch::NN::Linear.new(2, 3) }
    let(:sgd) { Torch::Optim::SGD.new(policy.parameters, lr: 0.1) }
    let(:torch_agent) { Reinforce::Algorithms::MonteCarloPolicyGradient.new(2, 3, 0.99, policy, sgd) }

    it "chooses an Integer action from a plain Array state" do
      # Regression: #choose_action used to pass CategoricalDistribution an
      # Array (logits.to_a) instead of a Tensor, crashing on the first step,
      # and returned a 0-dim tensor that environments' #step can't map onto
      # an action.
      TorchTestHelpers.with_torch_seed do
        action = torch_agent.choose_action([0.5, -0.5])

        expect(action.is_a?(Integer)).to be == true
        expect((0...3).cover?(action)).to be == true
      end
    end

    it "predicts without recording into the training buffer" do
      TorchTestHelpers.with_torch_seed do
        5.times do
          expect(torch_agent.predict([0.5, -0.5]).is_a?(Integer)).to be == true
        end

        expect(torch_agent.instance_variable_get(:@log_probs)).to be == []
      end
    end

    it "updates the policy parameters at the end of an episode" do
      # Regression: #update_policy never called loss.backward, so the
      # optimizer found no gradients and the policy never changed.
      TorchTestHelpers.with_torch_seed do
        snapshot = TorchTestHelpers.parameter_snapshot(policy)
        3.times do
          torch_agent.choose_action([0.5, -0.5])
          torch_agent.get_reward(1.0)
        end
        torch_agent.update_policy

        expect(TorchTestHelpers.parameters_changed?(policy, snapshot)).to be == true
      end
    end

    it "clears the episode buffers so consecutive episodes can each be learned from" do
      # Regression: the buffers were never cleared, so a second episode's
      # update would try to backpropagate through the first episode's
      # already-freed graph.
      TorchTestHelpers.with_torch_seed do
        2.times do
          torch_agent.choose_action([0.5, -0.5])
          torch_agent.get_reward(1.0)
          torch_agent.update_policy

          expect(torch_agent.instance_variable_get(:@log_probs)).to be == []
          expect(torch_agent.instance_variable_get(:@rewards)).to be == []
        end
      end
    end
  end
end
