# frozen_string_literal: true

require "reinforce"
require_relative "../../support/environment_contract"

describe Reinforce::Environments::MultiGoalGridWorld do
  # A 1x5 corridor (3x5 grid, bottom rows blocked) with goals at both ends.
  def corridor(**options)
    blocked = (1..2).flat_map { |row| (0..4).map { |column| [row, column] } }
    subject.new(5, [0, 1], [[0, 0], [0, 4]], blocked, **options)
  end

  include_context EnvironmentContract, factory: -> { Reinforce::Environments::MultiGoalGridWorld.four_rooms }

  it "reports which goal was reached, plus a step feature, and ends the episode at any goal" do
    env = corridor
    env.reset

    expect(env.step(:right)).to be == [[0, 2], 0.0, false, {features: [0.0, 0.0, 1.0]}]
    env.step(:right)
    expect(env.step(:right)).to be == [[0, 4], 0.0, true, {features: [0.0, 1.0, 1.0]}]
  end

  it "rewards features . task, and is reward-free by default" do
    env = corridor(task: [1.0, 2.0, -0.1])
    env.reset

    expect(env.step(:right)[1]).to be_within(1e-12).of(-0.1)
    env.reset
    expect(env.step(:left)[1]).to be_within(1e-12).of(0.9)
    expect(corridor.task).to be == [0.0, 0.0, 0.0]
  end

  it "can drop the step feature, and validates the task's size" do
    env = corridor(step_feature: false)
    env.reset

    expect(env.feature_size).to be == 2
    expect(env.step(:left)[3]).to be == {features: [1.0, 0.0]}
    expect { corridor(task: [1.0]) }.to raise_exception(ArgumentError)
    expect { subject.new(3, [0, 0], []) }.to raise_exception(ArgumentError)
  end

  it "computes the best achievable return, discounted or not" do
    env = corridor

    # Goal 0 is 1 step away, goal 1 three steps away.
    expect(env.best_return([1.0, 1.0, -0.1], max_steps: 10)).to be_within(1e-12).of(0.9)
    expect(env.best_return([0.0, 1.0, -0.1], max_steps: 10)).to be_within(1e-12).of(0.7)
    expect(env.best_return([0.0, 1.0, -0.1], max_steps: 2)).to be_within(1e-12).of(-0.1)
    expect(env.best_return([-1.0, -1.0, 0.0], max_steps: 10)).to be_within(1e-12).of(0.0)
    expect(env.best_return([0.0, 1.0, 0.0], max_steps: 10, discount_factor: 0.5)).to be_within(1e-12).of(0.25)
  end

  it "builds a four-rooms layout with one goal per room" do
    rooms = subject.four_rooms

    expect(rooms.goals).to be == [[4, 4], [0, 10], [10, 0], [10, 10]]
    expect(rooms.goals.all? { |goal| rooms.reachable_states.include?(goal) }).to be == true
    expect(rooms.feature_size).to be == 5
  end

  it "starts each episode at a random non-goal cell with exploring starts" do
    env = subject.four_rooms(random_start: true)
    starts = Array.new(200) { env.reset && env.state.dup }

    expect(starts.uniq.size).to be > 50
    expect(starts & env.goals).to be == []
    expect(starts.all? { |start| env.reachable_states.include?(start) }).to be == true
  end

  it "renders goals by index" do
    output = StringIO.new
    env = corridor
    env.reset
    env.render(output)

    expect(output.string.lines[1]).to be == "0 A _ _ 1 \n"
  end
end
