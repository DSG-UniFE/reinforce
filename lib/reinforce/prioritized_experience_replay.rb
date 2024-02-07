# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi.

require_relative './categorical_distribution'

module Reinforce
  ## This class provides the prioritized experience replay functionality, that
  #  is essential for some Reinforcement Learning algorithms, such as DQN.
  class PrioritizedExperienceReplay
    def initialize(buffer_size = 1000)
      @capacity = buffer_size
      @count = 0
      reset
    end

    def reset
      @pos = 0
      @count = 0
      @experience = no_experience
    end

    def size
      @count
    end

    # Here we want to avoid the buffer to grow indefinitely
    def bulk_update(experience)
    num_experiences = experience[:state].size
    num_experiences.times do |i|
      @pos = 0 if @pos >= @capacity

        @experience[:state][@pos] = experience[:state][i]
        @experience[:action][@pos] = experience[:action][i]
        @experience[:next_state][@pos] = experience[:next_state][i]
        @experience[:reward][@pos] = experience[:reward][i]
        @experience[:done][@pos] = experience[:done][i]

        @pos += 1
        @count = [@count + 1, @capacity].min
      end
    end

    def update(state, action, next_state, reward, done)
      # without a cap, the buffer will grow indefinitely
      @pos = 0 if @pos >= @capacity
      @experience[:state][@pos] = state
      @experience[:action][@pos] = action
      @experience[:next_state][@pos] = next_state
      @experience[:reward][@pos] = reward
      @experience[:done][@pos] = done
      
      @pos += 1 
      @count = [@count + 1, @capacity].min
    end

    def sample(size = 1)
      dist = CategoricalDistributionFactory.create_from_frequencies(@experience[:reward])
      size.times.map { dist.sample }.each_with_object(no_experience) do |idx, exp|
        exp[:state] << @experience[:state][idx]
        exp[:action] << @experience[:action][idx]
        exp[:next_state] << @experience[:next_state][idx]
        exp[:reward] << @experience[:reward][idx]
        exp[:done] << @experience[:done][idx]
      end
    end

    private

    def no_experience
      # Initialize the experience replay store.
      # For effeciency reasons, we allocate the memory for the experience now
      # and then we reuse it during the training process.
      { state: [], 
        action: [],#Array.new(@capacity), 
        next_state: [],#Array.new(@capacity), 
        reward: [], #Array.new(@capacity), 
        done: [] #Array.new(@capacity)
      }
    end
  end
end
