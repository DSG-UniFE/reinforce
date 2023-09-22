# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi.

module Reinforce
  class Experience
    def initialize
      reset
    end

    def reset
      @history = { state: [], action: [], next_state: [], reward: [], done: [] }
    end

    def history_size
      @history[:state].size
    end

    def update(state, action, next_state, reward, done)
      @history[:state] << state
      @history[:action] << action
      @history[:next_state] << next_state
      @history[:reward] << reward
      @history[:done] << done
    end

    def [](symbol)
      case symbol
      when :states
        states
      when :actions
        actions
      when :next_states
        next_states
      when :rewards
        rewards
      when :dones
        dones
      else
        raise ArgumentError, "Unknown symbol #{symbol}"
      end
    end

    def states
      @history[:next_stat]
    end

    def actions
      @history[:action]
    end

    def next_states
      @history[:next_state]
    end

    def rewards
      @history[:reward]
    end

    def dones
      @history[:done]
    end
  end
end
