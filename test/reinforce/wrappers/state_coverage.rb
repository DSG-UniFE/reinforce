# frozen_string_literal: true

require "reinforce"
require_relative "../../support/environment_contract"

describe Reinforce::Wrappers::StateCoverage do
  # A 1x3 corridor (3x3 grid with the bottom two rows blocked): start at the
  # left end, goal at the right end.
  def corridor
    Reinforce::Environments::GridWorld.new(3, [0, 0], [0, 2], [[1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]], reward: :sparse)
  end

  include_context EnvironmentContract,
    factory: -> { Reinforce::Wrappers::StateCoverage.new(Reinforce::Environments::GridWorld.new(3, [0, 0], [0, 2], [])) }

  it "counts distinct states across episodes, as a fraction of the reachable ones" do
    environment = subject.new(corridor)

    environment.reset
    environment.step(:right)
    environment.step(:left)
    expect([environment.states_visited, environment.steps_taken]).to be == [2, 2]
    expect(environment.coverage).to be_within(1e-12).of(2.0 / 3)

    environment.reset
    environment.step(:right)
    environment.step(:right) # reaches the goal: done

    expect(environment.states_visited).to be == 3
    expect(environment.coverage).to be_within(1e-12).of(1.0)
  end

  it "records coverage at the end of each episode, against total steps taken" do
    environment = subject.new(corridor)
    received = []
    environment.metrics.add_sink(->(key, value, step) { received << [key, value.round(3), step] })

    environment.reset
    environment.step(:right)
    environment.reset # closes the first episode
    environment.step(:right)
    environment.step(:right) # done closes the second

    expect(received).to be == [
      [:states_visited, 2, 1], [:state_coverage, 0.667, 1],
      [:states_visited, 3, 3], [:state_coverage, 1.0, 3]
    ]
  end

  it "only reports a count when the reachable states are unknown" do
    plain = Object.new
    plain.define_singleton_method(:reset) { [0] }
    plain.define_singleton_method(:step) { |_action| [[1], 0.0, true, {}] }
    environment = subject.new(plain)

    environment.reset
    environment.step(0)

    expect(environment.coverage).to be_nil
    expect(environment.metrics.keys).to be == [:states_visited]
  end

  it "accepts an explicit set of reachable states" do
    environment = subject.new(corridor, reachable_states: [[0, 0], [0, 1], [0, 2], [9, 9]])
    environment.reset

    expect(environment.coverage).to be_within(1e-12).of(0.25)
  end
end
