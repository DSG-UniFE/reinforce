# frozen_string_literal: true

require "reinforce/environment"

# A shared sus test context that checks a concrete environment satisfies
# the Reinforce::Environment contract (see lib/reinforce/environment.rb).
# Include it from an environment's own test file:
#
#   include_context EnvironmentContract, factory: -> { MyEnv.new(...) }
#
# `factory` must build and return a *fresh* environment each time it is
# called, since #step mutates environment state and these examples run
# independently of whatever the file's own `let(:environment)` does.
EnvironmentContract = Sus::Shared("behaves like a Reinforce::Environment") do |factory:|
  let(:contract_environment) { factory.call }

  it "includes Reinforce::Environment" do
    expect(contract_environment.class.ancestors.include?(Reinforce::Environment)).to be == true
  end

  it "responds to the required environment methods" do
    expect(contract_environment.respond_to?(:reset)).to be == true
    expect(contract_environment.respond_to?(:state_size)).to be == true
    expect(contract_environment.respond_to?(:actions)).to be == true
    expect(contract_environment.respond_to?(:step)).to be == true
  end

  it "has a state_size consistent with what #reset returns" do
    state = contract_environment.reset
    expect(state.size).to be == contract_environment.state_size
  end

  it "has a non-empty action space" do
    expect(contract_environment.actions.size > 0).to be == true
  end

  it "returns a (next_state, reward, done, info) 4-tuple from #step" do
    contract_environment.reset
    result = contract_environment.step(contract_environment.actions.first)

    expect(result.size).to be == 4
    _next_state, reward, done, info = result
    expect(reward.is_a?(Numeric)).to be == true
    expect([true, false].include?(done)).to be == true
    expect(info.is_a?(Hash)).to be == true
  end
end
