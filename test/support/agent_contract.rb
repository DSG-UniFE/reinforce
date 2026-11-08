# frozen_string_literal: true

require "reinforce/agent"
require_relative "torch_test_helpers"

# A shared sus test context that checks a concrete algorithm satisfies the
# Reinforce::Agent contract (see lib/reinforce/agent.rb): #predict returns
# an element of the environment's action space, greedily by default, and
# samples when called with `deterministic: false`. Include it from an
# algorithm's own test file:
#
#   include_context AgentContract,
#     factory: -> { MyAlgorithm.new(environment, ...) },
#     state: [0.0, 1.0],
#     actions: %i[left right]
#
# `factory` is evaluated with instance_exec inside each example, so it can
# use the including file's own `let`s and helper methods. `state` is any
# valid state for the agent, and `actions` its expected action space.
AgentContract = Sus::Shared("behaves like a Reinforce::Agent") do |factory:, state:, actions:|
  let(:contract_agent) { instance_exec(&factory) }

  it "includes Reinforce::Agent" do
    expect(contract_agent.class.ancestors.include?(Reinforce::Agent)).to be == true
  end

  it "predicts an element of the action space" do
    expect(actions).to be(:include?, contract_agent.predict(state))
  end

  it "predicts greedily by default, regardless of the random seed" do
    predictions = [1, 2, 3].map do |seed|
      TorchTestHelpers.with_torch_seed(seed) { contract_agent.predict(state) }
    end

    expect(predictions.uniq.size).to be == 1
  end

  it "samples from the action space when deterministic: false" do
    predictions = TorchTestHelpers.with_torch_seed do
      Array.new(50) { contract_agent.predict(state, deterministic: false) }
    end

    expect(predictions - actions).to be == []
    # An untrained agent's policy is close to uniform, so 50 draws that all
    # land on one action would mean #predict ignored `deterministic: false`.
    expect(predictions.uniq.size > 1).to be == true
  end
end
