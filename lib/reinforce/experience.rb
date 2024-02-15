# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi.

require 'forwardable'

module Reinforce
  class Experience
    extend Forwardable
    def_delegators :@history, :[]

    def initialize
      reset
    end

    def reset
      @history = { state: [], action: [], next_state: [], next_action: [], reward: [], done: [] }
    end

    def history_size
      @history[:state].size
    end

    def update(state, action, next_state, next_action=nil, reward, done)
      @history[:state] << state
      @history[:action] << action
      @history[:next_state] << next_state
      @history[:next_action] << next_action
      @history[:reward] << reward
      @history[:done] << done
    end

    def sample(size = 1)
      indices = (0...history_size).to_a.sample(size)
      {
        state: indices.map { |i| @history[:state][i] },
        action: indices.map { |i| @history[:action][i] },
        next_state: indices.map { |i| @history[:next_state][i] },
        next_action: indices.map { |i| @history[:next_action][i] },
        reward: indices.map { |i| @history[:reward][i] },
        done: indices.map { |i| @history[:done][i] }
      }
    end


    # def [](symbol)
    #   case symbol
    #   when :state
    #     states
    #   when :action
    #     actions
    #   when :next_state
    #     next_states
    #   when :reward
    #     rewards
    #   when :done
    #     dones
    #   else
    #     raise ArgumentError, "Unknown symbol #{symbol}"
    #   end
    # end

    # def states
    #   @history[:next_state]
    # end

    # def actions
    #   @history[:action]
    # end

    # def next_states
    #   @history[:next_state]
    # end

    # def rewards
    #   @history[:reward]
    # end

    # def dones
    #   @history[:done]
    # end
  end
end
