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
    class GridWorld
      include ::Reinforce::Environment

      ACTIONS = %i[up down left right].freeze
      attr_reader :state, :obstacles

      # @parameter obstacles [Integer, Array] either how many obstacles to
      #   place at random, distinct interior cells (never on `start` or
      #   `goal`), chosen once here; or an explicit Array of [row, column]
      #   obstacle positions.
      def initialize(size, start, goal, obstacles = 5)
        @size = size
        @start = start
        @goal = goal
        @obstacles = obstacles.is_a?(Array) ? obstacles.map(&:dup) : random_obstacles(obstacles)
        @state = @start.dup
      end

      def reset
        @state = @start.dup
      end

      def actions
        ACTIONS
      end

      def state_size
        @state.size
      end

      def step(action)
        action = ACTIONS[action] if action.is_a?(Integer)
        # warn "action: #{action}"
        next_state = @state.dup
        reward = 0
        done = false

        case action
        when :up
          next_state[0] -= 1 unless next_state[0].zero?
        when :down
          next_state[0] += 1 unless next_state[0] == @size - 1
        when :left
          next_state[1] -= 1 unless next_state[1].zero?
        when :right
          next_state[1] += 1 unless next_state[1] == @size - 1
        else
          raise "Invalid action: #{action}"
        end

        if @obstacles.include?(next_state)
          next_state = @state.dup  # Stay in the same state if moving into an obstacle
          reward = -1
        # if the agent did not move
        elsif next_state == @state
          reward = -1
        elsif next_state == @goal
          # warn "Goal reached!"
          done = true
          reward = 1
        end
        # Here we are trying to give the agent an incentive to move towards the distance_to_goal
        # otherwise, the agent will just stay in the same state and collect the reward
        distance_to_goal = (@goal[0] - next_state[0]).abs + (@goal[1] - next_state[1]).abs
        reward += -distance_to_goal if reward < 1

        @state = next_state

        [next_state, reward, done, {}]
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
