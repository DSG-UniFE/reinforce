#!/usr/bin/env ruby
# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Filippo Poltronieri, Mauro Tortonesi.

module Reinforce
  module Environments
    class Game2048
      ACTIONS = [:up, :down, :left, :right]

      attr_reader :board, :score

      def initialize(board_size = 4)
        @board_size = board_size
        reset
      end

      def reset
        @board = Array.new(@board_size) { Array.new(@board_size, 0) }
        @score = 0
        place_random_tile
        place_random_tile
        @board.flatten.dup
      end

      def actions
        ACTIONS
      end

      def state_size
        @board.flatten.length
      end

      def step(action)
        action = actions[action] if action.is_a?(Integer)
        reward = 0
        done = false

        current_state = @board.dup

        case action
        when :up, :down
          current_state = transpose(current_state) if action == :down
          current_state, reward = move_up(current_state)
          current_state = transpose(current_state) if action == :down
        when :left, :right
          current_state = reverse(current_state) if action == :right
          current_state, reward = move_left(current_state)
          current_state = reverse(current_state) if action == :right
        else
          raise "Invalid action: #{action}"
        end

        @score += reward

        #puts "action: #{action}"
        #self.render($stdout)
        if @board != current_state
          @board = current_state
        end

        # after the action is performed
        # let's add another random tile
        place_random_tile
        # check if the game is over
        done = game_over?
        [@board.flatten.dup, reward, done]
      end

      def render(output_stream)
        output_stream.puts '2048 Game Board:'
        @board.each { |row| output_stream.puts row.join("\t") }
        output_stream.puts "Score: #{@score}"
        output_stream.puts ''
      end

      private

      def place_random_tile
        empty_cells = []
        @board.each_with_index do |row, i|
          row.each_with_index do |cell, j|
            empty_cells << [i, j] if cell == 0
          end
        end

        return if empty_cells.empty?

        random_cell = empty_cells.sample
        @board[random_cell[0]][random_cell[1]] = [2, 4].sample
      end

      def transpose(matrix)
        matrix.transpose.map(&:dup)
      end

      def reverse(matrix)
        matrix.map(&:reverse)
      end

      def move_left(board)
        score = 0
        new_board = board.map do |row|
          merged = merge(row)
          score += merged[1]
          merged[0]
        end
        [new_board, score]
      end

      def move_up(board)
        transposed_board = transpose(board)
        new_board, score = move_left(transposed_board)
        [transpose(new_board), score]
      end

      def merge(row)
        score = 0
        new_row = row.dup

        # Combine adjacent tiles of the same value
        for i in 0...new_row.length - 1
          if new_row[i] == new_row[i + 1] && new_row[i] != 0
            new_row[i] *= 2
            new_row[i + 1] = 0
            score += new_row[i]
          end
        end

        # Shift non-zero tiles to the left
        new_row.reject! { |tile| tile == 0 }
        new_row += [0] * (row.length - new_row.length)

        [new_row, score]
      end

      def game_over?
        # Check if there are any valid moves left (tiles can be merged)
        valid_moves = false

        # before check if there are empty tiles
        @board.each do |row|
        end

        # Check horizontal moves (left and right)
        @board.each do |row|
          valid_moves = true if row.uniq == [0]
          if !valid_moves
            for i in 0...row.length - 1
              valid_moves = true if row[i] == row[i + 1] && row[i] != 0
            end
          end
        end

        # Check vertical moves (up and down)
        transposed_board = transpose(@board)
        transposed_board.each do |row|
          valid_moves = true if row.uniq == [0]
          if !valid_moves
            for i in 0...row.length - 1
              valid_moves = true if row[i] == row[i + 1] && row[i] != 0
            end
          end
        end

        !valid_moves
      end
    end
  end
end
