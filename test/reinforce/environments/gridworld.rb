# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi.

require 'reinforce/environments/gridworld'

describe Reinforce::Environments::GridWorld do
  let(:initial_state) do
    [0, 0]
  end

  let(:gridworld) do
    size = 5
    start = initial_state
    goal = [size - 1, size - 1]
    obstacles = [[1, 1], [2, 2], [3, 3]]
    Reinforce::Environments::GridWorld.new(size, start, goal, obstacles)
  end

  let(:gridworld_render) do
    <<~RENDER
      Gridworld:
      S _ _ _ _ 
      _ X _ _ _ 
      _ _ X _ _ 
      _ _ _ X _ 
      _ _ _ _ G 

    RENDER
  end

  it 'can be instantiated' do
    expect(gridworld).not.to be_nil
  end

  it 'can be moved and reset' do
    gridworld.reset
    next_state, = gridworld.step(:right)
    expect(gridworld.state).to be == next_state
    gridworld.reset
    expect(gridworld.state).to be == initial_state
  end

  it 'returns new state, reward, and completion status when moving' do
    gridworld.reset
    next_state, reward, done = gridworld.step(:right)
    expect(next_state).to be == [0, 1]
    expect(reward).to be == 0
    expect(done).to be_falsey
  end

  it 'can be rendered' do
    out = StringIO.new
    gridworld.render(out)
    expect(out.string).to be == gridworld_render
  end
end
