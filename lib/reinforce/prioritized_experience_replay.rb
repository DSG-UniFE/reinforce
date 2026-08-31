# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi.

require_relative "sum_tree"

module Reinforce
  ##
  # Prioritized Experience Replay (Schaul et al., 2015): samples transitions
  # with probability proportional to their TD-error magnitude instead of
  # uniformly, so learning spends more gradient steps on transitions the
  # current Q-function is most wrong about.
  #
  # Sampling probability is P(i) = priority_i^alpha / sum_j(priority_j^alpha)
  # (alpha trades off how much prioritization is used, 0 = uniform sampling,
  # 1 = fully greedy on priority). Because that oversamples high-priority
  # transitions, #sample also returns per-transition importance-sampling
  # weights (annealed via beta -> 1.0) that the caller must use to weight
  # its loss, correcting the bias so the expected update stays unbiased --
  # see DQN#train for the reference usage. A SumTree (sum_tree.rb) keeps
  # both sampling and priority updates O(log n).
  #
  # New transitions are inserted with the maximum priority seen so far (see
  # #update), so every transition is guaranteed to be trained on at least
  # once regardless of its actual TD-error; #update_priorities is how a
  # caller reports the real TD-error back after learning from a sampled
  # batch, so future sampling reflects it.
  class PrioritizedExperienceReplay
    def initialize(buffer_size = 1000, alpha: 0.6, beta: 0.4, beta_increment: 0.001, priority_epsilon: 0.01)
      @capacity = buffer_size
      @alpha = alpha
      @beta = beta
      @beta_increment = beta_increment
      @priority_epsilon = priority_epsilon
      reset
    end

    def reset
      @pos = 0
      @count = 0
      @max_priority = 1.0
      @tree = SumTree.new(@capacity)
      @experience = no_experience
    end

    def size
      @count
    end

    def update(state, action, next_state, reward, done, priority: @max_priority)
      store(@pos, state, action, next_state, reward, done, priority)
      advance
    end

    def bulk_update(experience)
      experience[:state].size.times do |i|
        store(@pos, experience[:state][i], experience[:action][i], experience[:next_state][i],
          experience[:reward][i], experience[:done][i], @max_priority)
        advance
      end
    end

    # Samples `size` transitions, stratified across `size` equal
    # priority-mass segments (the standard PER sampling scheme -- it
    # reduces the variance of the sample versus `size` independent draws).
    # Returns the sampled experience plus `indices` (to pass back into
    # #update_priorities) and `weights` (the importance-sampling
    # correction, already normalized so the batch's max weight is 1.0).
    def sample(size = 1)
      raise ArgumentError, "cannot sample from an empty buffer" if @count.zero?

      segment = @tree.total / size
      picks = size.times.map do |i|
        low = segment * i
        high = segment * (i + 1)
        cumulative_value = [low + (rand * (high - low)), @tree.total - Float::EPSILON].min
        @tree.get(cumulative_value)
      end

      sampled = picks.each_with_object(no_experience.merge(indices: [])) do |(data_index, _priority), exp|
        exp[:state] << @experience[:state][data_index]
        exp[:action] << @experience[:action][data_index]
        exp[:next_state] << @experience[:next_state][data_index]
        exp[:reward] << @experience[:reward][data_index]
        exp[:done] << @experience[:done][data_index]
        exp[:indices] << data_index
      end

      sampled[:weights] = importance_sampling_weights(picks)
      @beta = [@beta + @beta_increment, 1.0].min

      sampled
    end

    # Report the TD-errors observed for a batch previously returned by
    # #sample (matched positionally against its `indices`), updating each
    # transition's priority so subsequent sampling reflects it.
    def update_priorities(indices, td_errors)
      indices.each_with_index do |data_index, i|
        priority = td_errors[i].abs + @priority_epsilon
        @tree.update(data_index, priority**@alpha)
        @max_priority = [@max_priority, priority].max
      end
    end

    private

    def importance_sampling_weights(picks)
      weights = picks.map do |_data_index, priority|
        probability = priority / @tree.total
        (@count * probability)**(-@beta)
      end
      max_weight = weights.max
      weights.map { |weight| weight / max_weight }
    end

    def store(index, state, action, next_state, reward, done, priority)
      @experience[:state][index] = state
      @experience[:action][index] = action
      @experience[:next_state][index] = next_state
      @experience[:reward][index] = reward
      @experience[:done][index] = done
      @tree.update(index, priority**@alpha)
      @max_priority = [@max_priority, priority].max
    end

    def advance
      @pos += 1
      @pos = 0 if @pos >= @capacity
      @count = [@count + 1, @capacity].min
    end

    def no_experience
      # For efficiency reasons the arrays grow lazily (index assignment
      # extends them) rather than being preallocated to @capacity.
      {state: [], action: [], next_state: [], reward: [], done: []}
    end
  end
end
