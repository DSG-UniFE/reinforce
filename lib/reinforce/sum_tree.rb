# frozen_string_literal: true

module Reinforce
  ##
  # A binary tree where each leaf holds a priority and each internal node
  # holds the sum of its children's values. This makes both "update a leaf's
  # priority" and "find the leaf whose cumulative priority range contains a
  # given value" O(log n) operations, which is what lets
  # PrioritizedExperienceReplay sample transitions proportionally to
  # priority without scanning the whole buffer on every sample (Schaul et
  # al., 2015, appendix B.2.1).
  class SumTree
    def initialize(capacity)
      @capacity = capacity
      @tree = Array.new(2 * capacity - 1, 0.0)
    end

    # The sum of every leaf's priority, i.e. the root of the tree.
    def total
      @tree[0]
    end

    def update(data_index, priority)
      tree_index = data_index + @capacity - 1
      change = priority - @tree[tree_index]
      @tree[tree_index] = priority
      propagate(tree_index, change)
    end

    # Given a value in [0, total), walks down from the root and returns the
    # [data_index, priority] of the leaf whose cumulative priority range
    # contains it -- the standard stratified-sampling trick that turns a
    # single random draw into a priority-proportional pick in O(log n).
    def get(cumulative_value)
      tree_index = 0

      loop do
        left = 2 * tree_index + 1
        break if left >= @tree.length

        right = left + 1
        if cumulative_value <= @tree[left]
          tree_index = left
        else
          cumulative_value -= @tree[left]
          tree_index = right
        end
      end

      [tree_index - @capacity + 1, @tree[tree_index]]
    end

    private

    def propagate(tree_index, change)
      return if tree_index.zero?

      parent = (tree_index - 1) / 2
      @tree[parent] += change
      propagate(parent, change)
    end
  end
end
