# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi.

require_relative './categorical_distribution'

module Reinforce
  ## This class provides the prioritized experience replay functionality, that
  #  is essential for some Reinforcement Learning algorithms, such as DQN.
  class PrioritizedExperienceReplay
    def initialize
      reset
    end

    def reset
      @experience = { state: [], action: [], next_state: [], reward: [], done: [] }
    end

    def size
      @experience[:state].size
    end

    def bulk_update(experience)
      @experience[:state] += experience[:state]
      @experience[:action] += experience[:action]
      @experience[:next_state] += experience[:next_state]
      @experience[:reward] += experience[:reward]
      @experience[:done] += experience[:done]
    end

    def update(state, action, next_state, reward, done)
      @experience[:state] << state
      @experience[:action] << action
      @experience[:next_state] << next_state
      @experience[:reward] << reward
      @experience[:done] << done
    end

    def sample(size = 1)
      dist = CategoricalDistributionFactory.create_from_frequencies(@experience[:reward])
      ret = size.times.map do
        idx = dist.sample
        [@experience[:state][idx],
         @experience[:action][idx],
         @experience[:next_state][idx],
         @experience[:reward][idx],
         @experience[:done][idx]]
      end
      size == 1 ? ret.first : ret
    end
  end
end
