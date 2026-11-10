# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi.

require "reinforce/environments/gridworld"
require_relative "../../support/environment_contract"

describe Reinforce::Environments::GridWorld do
  include_context EnvironmentContract, factory: -> { Reinforce::Environments::GridWorld.new(5, [0, 0], [4, 4], [[1, 1], [2, 2], [3, 3]]) }

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

  it "can be instantiated" do
    expect(gridworld).not.to be_nil
  end

  it "can be moved and reset" do
    gridworld.reset
    next_state, = gridworld.step(:right)
    expect(gridworld.state).to be == next_state
    gridworld.reset
    expect(gridworld.state).to be == initial_state
  end

  it "returns new state, reward, and completion status when moving" do
    gridworld.reset
    next_state, reward, done = gridworld.step(:right)
    expect(next_state).to be == [0, 1]
    expect(reward).to be == -((4 - next_state[0]).abs + (4 - next_state[1]).abs)
    expect(done).to be_falsey
  end

  it "uses explicitly given obstacle positions, unchanged across resets" do
    gridworld.reset
    gridworld.reset

    expect(gridworld.obstacles).to be == [[1, 1], [2, 2], [3, 3]]
  end

  describe "with randomly placed obstacles" do
    let(:random_gridworld) { Reinforce::Environments::GridWorld.new(5, [0, 0], [4, 4], 4) }

    it "keeps the same obstacles across resets" do
      # Regression: #reset re-randomized the obstacles, but the state is only
      # the agent's position, so the agent could never see where they were.
      obstacles = random_gridworld.obstacles.dup
      3.times { random_gridworld.reset }

      expect(random_gridworld.obstacles).to be == obstacles
    end

    it "places the requested number of distinct interior obstacles, off start and goal" do
      20.times do
        obstacles = Reinforce::Environments::GridWorld.new(5, [0, 0], [4, 4], 4).obstacles

        expect(obstacles.uniq.size).to be == 4
        expect(obstacles.all? { |row, column| (1..3).cover?(row) && (1..3).cover?(column) }).to be == true
        expect(obstacles & [[0, 0], [4, 4]]).to be == []
      end
    end

    it "refuses more obstacles than there are free interior cells" do
      expect { Reinforce::Environments::GridWorld.new(4, [0, 0], [3, 3], 5) }.to raise_exception(ArgumentError)
    end
  end

  it "can be rendered" do
    out = StringIO.new
    gridworld.render(out)
    expect(out.string).to be == gridworld_render
  end

  describe "sparse reward" do
    let(:sparse) { Reinforce::Environments::GridWorld.new(3, [0, 0], [0, 2], [], reward: :sparse) }

    it "gives 0 for ordinary moves and bumps, and 1 for reaching the goal" do
      sparse.reset

      expect(sparse.step(:up)[1]).to be == 0
      expect(sparse.step(:right)[1]).to be == 0
      _state, reward, done = sparse.step(:right)
      expect([reward, done]).to be == [1, true]
    end

    it "keeps the dense reward by default" do
      dense = Reinforce::Environments::GridWorld.new(3, [0, 0], [0, 2], [])
      dense.reset

      expect(dense.step(:right)[1]).to be == -1
    end

    it "rejects unknown reward schemes" do
      expect { Reinforce::Environments::GridWorld.new(3, [0, 0], [0, 2], [], reward: :shaped) }.to raise_exception(ArgumentError)
    end
  end

  describe ".four_rooms" do
    let(:rooms) { Reinforce::Environments::GridWorld.four_rooms(size: 11) }

    it "builds walls through the middle with one doorway per wall segment, sparse by default" do
      expect(rooms.reward).to be == :sparse
      expect(rooms.obstacles.size).to be == 21 - 4
      [[2, 5], [8, 5], [5, 2], [5, 8]].each { |doorway| expect(rooms.obstacles).not.to be(:include?, doorway) }
      expect(rooms.obstacles).to be(:include?, [5, 5])
      expect([rooms.start, rooms.goal]).to be == [[0, 0], [10, 10]]
    end

    it "keeps every free cell reachable from the start, goal included" do
      expect(rooms.reachable_states.size).to be == 11 * 11 - rooms.obstacles.size
      expect(rooms.reachable_states).to be(:include?, rooms.goal)
    end

    it "rejects sizes too small for four rooms" do
      expect { Reinforce::Environments::GridWorld.four_rooms(size: 4) }.to raise_exception(ArgumentError)
    end
  end

  it "finds the reachable states around obstacles" do
    # The start's corner is sealed off by two obstacles.
    sealed = Reinforce::Environments::GridWorld.new(3, [0, 0], [2, 2], [[0, 1], [1, 0]])

    expect(sealed.reachable_states).to be == [[0, 0]]
  end

  describe "one-hot observations" do
    let(:one_hot) { Reinforce::Environments::GridWorld.new(3, [0, 0], [2, 2], [[1, 1]], observation: :one_hot) }

    include_context EnvironmentContract,
      factory: -> { Reinforce::Environments::GridWorld.new(3, [0, 0], [2, 2], [], observation: :one_hot) }

    it "observes a size * size vector with a 1.0 at the agent's cell, tracking its position internally" do
      expect(one_hot.state_size).to be == 9
      expect(one_hot.reset).to be == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

      next_observation, _reward, _done = one_hot.step(:down)

      expect(next_observation).to be == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
      expect(one_hot.state).to be == [1, 0]
    end

    it "leaves rewards, obstacles and reachability unchanged" do
      coordinates = Reinforce::Environments::GridWorld.new(3, [0, 0], [2, 2], [[1, 1]])
      [one_hot, coordinates].each(&:reset)

      %i[right down down right].each do |action|
        expect(one_hot.step(action)[1..2]).to be == coordinates.step(action)[1..2]
      end
      expect(one_hot.reachable_states).to be == coordinates.reachable_states
    end

    it "is available for four rooms too" do
      rooms = Reinforce::Environments::GridWorld.four_rooms(size: 5, observation: :one_hot)

      expect(rooms.reset.size).to be == 25
    end

    it "rejects unknown observation formats" do
      expect { Reinforce::Environments::GridWorld.new(3, [0, 0], [2, 2], [], observation: :pixels) }.to raise_exception(ArgumentError)
    end
  end
end
