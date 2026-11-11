# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi and Filippo Poltronieri.

module Reinforce
  module Environments
    # A size x size grid: move from `start` to `goal`, avoiding obstacles.
    #
    # The obstacles are fixed for the lifetime of the environment. The state
    # is only the agent's position, so obstacles that moved on every #reset
    # (as they used to) were invisible to the agent: a deterministic policy
    # could push into one forever, and only a sampled policy could get
    # around it. Create a new GridWorld for a different layout.
    #
    # Two reward schemes:
    #
    # - :dense (the default): every step costs the Manhattan distance to the
    #   goal (plus 1 for bumping into a wall or obstacle), and reaching the
    #   goal gives +1. The reward points towards the goal everywhere, so
    #   exploration is easy -- good for checking that an algorithm learns.
    # - :sparse: +1 for reaching the goal, 0 otherwise. Nothing points the
    #   way, so the agent has to find the goal by exploring -- the setting
    #   exploration bonuses and reward-free methods are meant for. See also
    #   GridWorld.four_rooms.
    #
    # Two observation formats (the environment tracks the agent's position
    # internally either way; #state, #start, #goal, #obstacles and
    # #reachable_states are always [row, column] positions):
    #
    # - :coordinates (the default): the [row, column] position itself.
    # - :one_hot: a size * size vector of 0.0s with a 1.0 at the agent's
    #   cell. Lets a small network treat every cell independently -- from
    #   raw coordinates it has to express "go right, except in this one
    #   cell next to a doorway" as a smooth function, which it learns
    #   poorly, so greedy policies tend to walk into walls.
    class GridWorld
      include ::Reinforce::Environment

      ACTIONS = %i[up down left right].freeze
      REWARDS = %i[dense sparse].freeze
      OBSERVATIONS = %i[coordinates one_hot].freeze
      attr_reader :state, :obstacles, :size, :start, :goal, :reward, :observation

      # A size x size grid split into four rooms by a horizontal and a
      # vertical wall through its middle, with one doorway in each of the
      # four wall segments (the classic "four rooms" exploration domain).
      # Start and goal default to opposite corners, so reaching the goal
      # means passing through two doorways. Sparse reward by default.
      #
      # @parameter size [Integer] an odd number of at least 5 keeps the
      #   rooms the same size.
      def self.four_rooms(size: 11, start: [0, 0], goal: [size - 1, size - 1], reward: :sparse, observation: :coordinates)
        raise ArgumentError, "four rooms needs a size of at least 5" if size < 5

        middle = size / 2
        doorways = [[size / 4, middle], [middle + (size - middle) / 2, middle], [middle, size / 4], [middle, middle + (size - middle) / 2]]
        walls = (0...size).map { |row| [row, middle] } + (0...size).map { |column| [middle, column] }
        new(size, start, goal, walls.uniq - doorways, reward:, observation:)
      end

      # @parameter obstacles [Integer, Array] either how many obstacles to
      #   place at random, distinct interior cells (never on `start` or
      #   `goal`), chosen once here; or an explicit Array of [row, column]
      #   obstacle positions.
      # @parameter reward [Symbol] :dense or :sparse (see above).
      # @parameter observation [Symbol] :coordinates or :one_hot (see above).
      def initialize(size, start, goal, obstacles = 5, reward: :dense, observation: :coordinates)
        raise ArgumentError, "reward must be one of #{REWARDS.inspect}, got #{reward.inspect}" unless REWARDS.include?(reward)
        unless OBSERVATIONS.include?(observation)
          raise ArgumentError, "observation must be one of #{OBSERVATIONS.inspect}, got #{observation.inspect}"
        end

        @size = size
        @start = start
        @goal = goal
        @reward = reward
        @observation = observation
        @obstacles = obstacles.is_a?(Array) ? obstacles.map(&:dup) : random_obstacles(obstacles)
        @state = @start.dup
      end

      # Every cell reachable from `start` (including `start` and, if
      # reachable, `goal`), found by breadth-first search around the
      # obstacles: the denominator for state-coverage measurements.
      #
      # @returns [Array(Array(Integer))] [row, column] positions.
      def reachable_states
        @reachable_states ||= begin
          blocked = @obstacles.to_set
          seen = {@start => true}
          frontier = [@start]
          until frontier.empty?
            row, column = frontier.shift
            [[row - 1, column], [row + 1, column], [row, column - 1], [row, column + 1]].each do |cell|
              next unless cell.all? { |coordinate| coordinate.between?(0, @size - 1) }
              next if blocked.include?(cell) || seen[cell]

              seen[cell] = true
              frontier << cell
            end
          end
          seen.keys
        end
      end

      def reset
        @state = @start.dup
        observe(@state)
      end

      def actions
        ACTIONS
      end

      def state_size
        (@observation == :one_hot) ? @size * @size : @state.size
      end

      # The observation for a [row, column] position, in this environment's
      # observation format.
      def observe(position)
        return position.dup unless @observation == :one_hot

        vector = Array.new(@size * @size, 0.0)
        vector[position[0] * @size + position[1]] = 1.0
        vector
      end

      # Where `action` leads from `position`: one cell in its direction,
      # unless that's off the grid or an obstacle, in which case the agent
      # stays put.
      #
      # @returns [Array(Array(Integer), Boolean)] the next position, and
      #   whether the agent bumped into the border or an obstacle.
      def move(position, action)
        action = ACTIONS[action] if action.is_a?(Integer)
        next_position = position.dup
        case action
        when :up
          next_position[0] -= 1 unless next_position[0].zero?
        when :down
          next_position[0] += 1 unless next_position[0] == @size - 1
        when :left
          next_position[1] -= 1 unless next_position[1].zero?
        when :right
          next_position[1] += 1 unless next_position[1] == @size - 1
        else
          raise "Invalid action: #{action}"
        end
        next_position = position.dup if @obstacles.include?(next_position)

        [next_position, next_position == position]
      end

      def step(action)
        next_state, bumped = move(@state, action)
        reward = 0
        done = false

        if bumped
          reward = -1
        elsif next_state == @goal
          done = true
          reward = 1
        end

        if @reward == :sparse
          reward = done ? 1 : 0
        elsif reward < 1
          # Dense: an incentive to move towards the goal, otherwise the
          # agent could just stay put.
          distance_to_goal = (@goal[0] - next_state[0]).abs + (@goal[1] - next_state[1]).abs
          reward += -distance_to_goal
        end

        @state = next_state

        [observe(next_state), reward, done, {}]
      end

      def render(output_stream)
        output_stream.puts "Gridworld:"
        (0...@size).each do |i|
          line = ""
          (0...@size).each do |j|
            line += if @start == [i, j]
              "S "
            elsif @goal == [i, j]
              "G "
            elsif @obstacles.include?([i, j])
              "X "
            elsif @state == [i, j]
              "A "
            else
              "_ "
            end
          end
          output_stream.puts line
        end
        output_stream.puts ""
      end

      private

      # Interior cells only (not on the grid's border), as before: with
      # start and goal on the border, the border stays a free path between
      # them however the obstacles fall.
      def random_obstacles(count)
        interior = (1..(@size - 2)).to_a.product((1..(@size - 2)).to_a) - [@start, @goal]
        raise ArgumentError, "cannot place #{count} obstacles in #{interior.size} free interior cells" if count > interior.size

        interior.sample(count)
      end
    end
  end
end
