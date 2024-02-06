#!/usr/bin/env ruby
# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi, Filippo Poltronieri.
#
module Reinforce
  module Environments
    class Game2048
      attr_reader :board,:score

      def initialize(board_size = 4)
        @board_size = board_size
        @board = Array.new(@board_size) { Array.new(@board_size) {0} }
        @score = 0
        place_random_tile
        @board.flatten.dup
      end

      def reset
        @done = false
        @board = Array.new(@board_size) { Array.new(@board_size) {0} }
        @score = 0
        place_random_tile
        @board.flatten.dup
      end

      def actions
        [:up, :down, :left, :right]
      end

      def state_size
        @board.flatten.length
      end

      def step(action)
				action = actions[action] if action.is_a?(Integer)
        reward = 0
        #done = false

				current_state = @board.dup

        case action
        when :up, :down
          #warn "current_state before: #{current_state}"
          current_state = transpose(current_state) 
        when :right, :down
          current_state = reverse(current_state)
        end

        reward = 0
        @board_size.times do |i|
          # here I wan to remove 0 values
          a = current_state[i].reject{|k| k == 0}
          #warn "a: #{a}"
          @board_size.times do |x|
            if a[x].to_i == a[x + 1]
              a[x], a[x + 1] = a[x] * 2, 0
              reward += a[x]
            end
          end
            current_state[i] = a.reject{|k| k == 0}.concat([0] * 4)[0..3]
        end
        current_state = reverse(current_state) if [:right, :down].include?(action)
        current_state = transpose(current_state) if [:up, :down].include?(action)
             
        # increase the score only for positive rewardsi
        # e.g the action actually merged some tiles
        if reward > 0
          @score += reward
        else
          # penalize the move with a negative rewards
          # e.g. the action did not merge any tiles
          # this is done to avoid the agent to do useless moves
          # and to force it to merge tiles
          # the reward is -1
          reward = -1
        end

        if current_state != @board 
          @board = current_state.dup
        else
          reward -= 20
        end
        # after the action is performed
        # let's add another random tile
        # check if the game is over
        #unless @board.flatten.include?(0)
        #  @done = true
        #  warn "Episode score #{@score}"
        #else 
        #@done = false 
        place_random_tile
        #end

        [@board.flatten.dup, reward, @done]
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

        unless empty_cells.empty?
          random_cell = empty_cells.sample
          @board[random_cell[0]][random_cell[1]] = [2, 4].sample
        end
        check_game_over
      end

      def can_merge_tiles?
        # Check for horizontal merges
        @board.each do |row|
          row.each_cons(2) do |a, b|
            return true if a == b
          end
        end
        # Check for vertical merges
        @board.transpose.each do |col|
          col.each_cons(2) do |a, b|
            return true if a == b
          end
        end
      
        false # No possible merges found
      end

      # Add this method to check for game over condition
      def check_game_over
        if !@board.flatten.include?(0) && !can_merge_tiles?
          @done = true
          warn "Game Over. Final score: #{@score}"
        else
          @done = false
        end
      end

      def transpose(matrix)
        matrix.transpose.map(&:dup)
      end

      def reverse(matrix)
        matrix.map(&:reverse)
      end

   end
  end
end
