# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi.

module Reinforce
  module Environments
    class GridWorld
      ACTIONS = %i[up down left right].freeze
      attr_reader :state

      def initialize(size, start, goal, obstacles)
        @size = size
        @start = start
        @goal = goal
        @obstacles = obstacles
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
        next_state = @state.dup
        reward = 0
        done = false

        case action
        when :up
          next_state[0] -= 1 unless (next_state[0]).zero?
        when :down
          next_state[0] += 1 unless next_state[0] == @size - 1
        when :left
          next_state[1] -= 1 unless (next_state[1]).zero?
        when :right
          next_state[1] += 1 unless next_state[1] == @size - 1
        else
          raise "Invalid action: #{action}"
        end

        if @obstacles.include?(next_state)
          next_state = @state.dup  # Stay in the same state if moving into an obstacle
          reward = -1
        elsif next_state == @goal
          done = true
          reward = 1
        end
        # Here we are trying to give the agent an incentive to move towards the distance_to_goal
        # otherwise, the agent will just stay in the same state and collect the reward
        distance_to_goal = (@goal[0] - next_state[0]).abs + (@goal[1] - next_state[1]).abs
        reward = -distance_to_goal if reward < 1

        @state = next_state

        [next_state, reward, done]
      end

      def render(output_stream)
        output_stream.puts 'Gridworld:'
        (0...@size).each do |i|
          line = ''
          (0...@size).each do |j|
            line += if @start == [i, j]
                      'S '
                    elsif @goal == [i, j]
                      'G '
                    elsif @obstacles.include?([i, j])
                      'X '
                    else
                      '_ '
                    end
          end
          output_stream.puts line
        end
        output_stream.puts ''
      end
    end
  end
end
