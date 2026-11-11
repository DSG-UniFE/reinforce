# frozen_string_literal: true

require_relative "gridworld"

module Reinforce
  module Environments
    # A GridWorld with several goals and a family of tasks over them, for
    # successor features and other multi-task methods. Each transition has a
    # feature vector -- reported in `info[:features]` --
    #
    #   [reached goal 0?, reached goal 1?, ..., reached goal n-1?, 1.0]
    #
    # (the trailing 1.0 is a per-step feature, so a task can put a price on
    # time; pass `step_feature: false` to drop it). The episode ends when any
    # goal is reached. A task is a weight vector `w` over those features, and
    # the reward is features . w: e.g. w = [1, 0, 0, 0, -0.01] means "reach
    # goal 0, each step costs 0.01", and [1, -1, 0.5, 0, -0.01] mixes
    # preferences. The default task is all zeros, so unless one is given the
    # environment is reward-free and only the features carry information.
    class MultiGoalGridWorld < GridWorld
      attr_reader :goals
      attr_accessor :task

      # A four-rooms layout (see GridWorld.four_rooms) with one goal per
      # room: the start room's inner corner and the other rooms' outer
      # corners.
      def self.four_rooms(size: 11, start: [0, 0], task: nil, step_feature: true, observation: :coordinates, random_start: false)
        rooms = GridWorld.four_rooms(size:, start:)
        middle = size / 2
        goals = [[middle - 1, middle - 1], [0, size - 1], [size - 1, 0], [size - 1, size - 1]]
        new(size, start, goals, rooms.obstacles, task:, step_feature:, observation:, random_start:)
      end

      # @parameter goals [Array(Array(Integer))] [row, column] goal cells.
      # @parameter task [Array(Numeric), nil] weights over the features
      #   (zeros by default: reward-free).
      # @parameter random_start [Boolean] start each episode at a uniformly
      #   random reachable cell that isn't a goal ("exploring starts"),
      #   instead of at `start`. Lets off-policy learners see every goal
      #   during training even when undirected exploration from `start`
      #   would almost always end at the nearest one.
      def initialize(size, start, goals, obstacles = [], task: nil, step_feature: true, observation: :coordinates, random_start: false)
        raise ArgumentError, "at least one goal is required" if goals.empty?

        super(size, start, goals.first, obstacles, observation:)
        @goals = goals.map(&:dup)
        @step_feature = step_feature
        @random_start = random_start
        @task = task || Array.new(feature_size, 0.0)
        raise ArgumentError, "task needs #{feature_size} weights" unless @task.size == feature_size
      end

      # @returns [Integer] the length of each transition's feature vector.
      def feature_size
        @goals.size + (@step_feature ? 1 : 0)
      end

      def reset
        return super unless @random_start

        @state = (reachable_states - @goals).sample.dup
        observe(@state)
      end

      def step(action)
        @state, = move(@state, action)
        features = features_for(@state)
        reward = features.zip(@task).sum { |feature, weight| feature * weight }
        [observe(@state), reward, @goals.include?(@state), {features:}]
      end

      # The best return achievable on `task` within `max_steps` steps from
      # the start, discounted by `discount_factor` (1.0: undiscounted): reach
      # the goal k that maximizes
      #
      #   task[k] * gamma^(d_k - 1) + (step weight) * sum_{t < d_k} gamma^t
      #
      # where d_k is the shortest path to k that doesn't pass through
      # another goal, or -- if every goal is worth less than wandering --
      # never reach one. The yardstick for zero-shot transfer; compare with
      # the discount factor the agent optimizes, since discounting can
      # change which goal is best.
      def best_return(task, max_steps:, discount_factor: 1.0)
        step_weight = @step_feature ? task.last : 0.0
        steps_value = ->(count) { (0...count).sum { |t| discount_factor**t } }
        candidates = goal_distances.filter_map do |goal_index, distance|
          next if distance > max_steps

          task[goal_index] * discount_factor**(distance - 1) + step_weight * steps_value.call(distance)
        end
        (candidates + [step_weight * steps_value.call(max_steps)]).max
      end

      def render(output_stream)
        output_stream.puts "MultiGoalGridWorld:"
        (0...@size).each do |row|
          line = (0...@size).map do |column|
            cell = [row, column]
            if @state == cell then "A "
            elsif (index = @goals.index(cell)) then "#{index} "
            elsif @obstacles.include?(cell) then "X "
            elsif @start == cell then "S "
            else "_ "
            end
          end
          output_stream.puts line.join
        end
        output_stream.puts ""
      end

      private

      def features_for(position)
        features = Array.new(@goals.size, 0.0)
        index = @goals.index(position)
        features[index] = 1.0 if index
        features << 1.0 if @step_feature
        features
      end

      # Shortest path length from the start to each goal reachable without
      # passing through another goal (goals end the episode).
      def goal_distances
        distances = {}
        seen = {@start => 0}
        frontier = [@start]
        until frontier.empty?
          position = frontier.shift
          ACTIONS.each do |action|
            next_position, bumped = move(position, action)
            next if bumped || seen.key?(next_position)

            seen[next_position] = seen[position] + 1
            if (index = @goals.index(next_position))
              distances[index] = seen[next_position]
            else
              frontier << next_position
            end
          end
        end
        distances
      end
    end
  end
end
