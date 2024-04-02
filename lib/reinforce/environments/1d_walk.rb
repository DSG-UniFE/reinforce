# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi and Filippo Poltronieri.


module Reinforce
  module Environments
    class OneDimensionalWalk
    ACTIONS = %i[left right].freeze
    attr_reader :state

    # Initialize the environment
    def initialize(size)
      @size = size
      @start = nil
      @goal = nil
      reset
    end

    def reset
      @start = rand(@size)
      @goal = rand(@size)
      @state = [@start, @goal].dup
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
      when :left
        next_state[0] -= 1 unless (next_state[0]).zero?
      when :right
        next_state[0] += 1 unless next_state[0] == @size - 1
      else
        raise "Invalid action: #{action}"
      end
      if next_state[0] == @goal
        reward = 1
        done = true
      end
      @state = next_state
      [next_state, reward, done]
    end

    def render(output)
      @size.each do |i|
        if i == @state[0]
          output.print 'A'
        elsif i == @goal
          output.print 'G'
        else
        output.print '.'
        end
      end
    end

  end
  end

end

