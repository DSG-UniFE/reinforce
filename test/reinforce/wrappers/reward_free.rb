# frozen_string_literal: true

require "reinforce"
require_relative "../../support/environment_contract"

describe Reinforce::Wrappers::RewardFree do
  def gridworld
    Reinforce::Environments::GridWorld.new(3, [0, 0], [0, 1], [], reward: :sparse)
  end

  include_context EnvironmentContract,
    factory: -> { Reinforce::Wrappers::RewardFree.new(Reinforce::Environments::GridWorld.new(3, [0, 0], [0, 1], [])) }

  it "hides the reward, keeping it in the step info, and leaves termination unchanged" do
    environment = subject.new(gridworld)
    environment.reset

    next_state, reward, done, info = environment.step(:right)

    expect([next_state, reward, done]).to be == [[0, 1], 0.0, true]
    expect(info).to be == {extrinsic_reward: 1}
  end

  it "passes other methods through to the wrapped environment" do
    environment = subject.new(gridworld)

    expect(environment.reachable_states.size).to be == 9
    expect(environment.respond_to?(:obstacles)).to be == true
    expect { environment.no_such_method }.to raise_exception(NoMethodError)
  end
end
