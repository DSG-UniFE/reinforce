# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi and Filippo Poltronieri.

module Reinforce
  module Environments
    # This is a modified version of Gridworld suitable for DeepSets
    # In this environment, the agent can move and pick up objects
    # And finally reach the goal
    # The action space is of dimension 5 (up, down, left, right, pick)
    class DSGridWorld
      ACTIONS = %i[up down left right].freeze
      attr_reader :state
      #srand(0)
      # We want to define a state which is compatible with deep sets
      # So we need to definea state as a list of objects and the agent position
      # Each vector  will represent the item type (agent_position, goal, item) and the position
      # we use an integer to represent the item type: 0 -> agent_position, 1 -> goal
      # The position is represented as a vector of two elements
      # The state is a list of vectors
      # The position of the agent is represented as following 
      # e.g [0, 2, 3] -> agent is at position 2,3. 0 means the agent
      # goal: [1, 4, 5] -> goal is at position 4,5. 1 it's the goal
      # obstacles: [2, 6, 7] -> 2 is an obstacles is at position 6,7.
      def initialize(size, start, goal, obstacles = 5)
        @size = size
        @start = start
        @goal = goal
        @obstacles_num = obstacles
        @obstacles = nil
        @state = nil
        @max_steps = 512
        @dstep = 0
        reset
      end

      def reset
        #warn "Resetting the environment after #{@step} steps."
        @dstep = 0
        @picked_objects = 0
        @obstacles = Array.new(@obstacles_num) { |_| [1 + rand(@size - 2), 1 + rand(@size - 2)] }
        encode_state
      end

      def encode_state
        @state = []
        @state << [0, @start[0], @start[1]].map(&:to_f)
        @state << [1, @goal[0], @goal[1]].map(&:to_f)
        @obstacles.each do |obj|
          @state << [2, obj[0], obj[1]].map(&:to_f)
        end 
        @state.clone
      end

      def actions
        ACTIONS
      end

      def action_masks
        masks = Array.new(ACTIONS.size, 1)
        to_fill = state_size[0] - masks.size
        to_fill.times { masks << 0 } 
        # Check if an action is possible in a given state
        agent_position = @state.select { |item| item[0] == 0 }
        agent_position = agent_position[0] if agent_position != []
        if agent_position[1] == 0
          masks[0] = 0
        end
        if agent_position[1] == (@size - 1)
          masks[1] = 0
        end
        if agent_position[2] == 0
          masks[2] = 0
        end
        if agent_position[2] == (@size - 1)
          masks[3] = 0
        end
        # Check also if is about to reach an obstacle position 
        # if there obstacles to the right, left, up or down
        @obstacles.each do |obj|
          if agent_position[1] == obj[0] && agent_position[2] == obj[1] - 1
            masks[3] = 0
          end
          if agent_position[1] == obj[0] && agent_position[2] == obj[1] + 1
            masks[2] = 0
          end
          if agent_position[1] == obj[0] - 1 && agent_position[2] == obj[1]
            masks[1] = 0
          end
          if agent_position[1] == obj[0] + 1 && agent_position[2] == obj[1]
            masks[0] = 0
          end
        end
        masks
      end 

      def state_size
        [@state.size, @state[0].size]
      end

      def step(action)
        action = ACTIONS[action] if action.is_a?(Integer)
        @dstep += 1
        # The state is encoded so we need to decode it to get the agent position, which can be any of the vectors in the state
        # Because we are using DeepSets. Let's check the vector with the item type 0
        agent_position = @state.select { |item| item[0] == 0 }
        agent_position = agent_position[0] if agent_position != []
        current_position = agent_position.dup

        # the element at position 1 and 2 are the x and y coordinates of the agent
        # There are no obstacles in this simple implementation so let's assume the agent can move freely within the grid world
        reward = -1
        done = false
        case action
        when :up
          agent_position[1] -= 1 unless agent_position[1] == 0.0
        when :down
          agent_position[1] += 1 unless agent_position[1] == (@size - 1)
        when :left
          agent_position[2] -= 1 unless (agent_position[2]) == 0.0
        when :right
          agent_position[2] += 1 unless agent_position[2] == (@size - 1)
        else
          #reward = -1E8 # a sort of action mapping here
        end
        
        # check if the agent reached a terminal state (the goal position)
        # the agent should maximize the number of objects picked up but also reach the goal
        goal_position = @state.select { |item| item[0] == 1 }
        goal_position = goal_position[0] if goal_position != []
        # if agent move to an obstacle position, move back the agent and assign a penalty
        
        @obstacles.each do |obj|
          if agent_position[1] == obj[0] && agent_position[2] == obj[1]
            agent_position[1] = current_position[1]
            agent_position[2] = current_position[2]
            reward = -10
          end
        end

        if agent_position[1] == goal_position[1] && agent_position[2] == goal_position[2]
          reward = 10
          done = true
          puts "Goal reached! in #{@dstep} steps"
        end


        if done != true && @dstep >= @max_steps
          done = true
        end 
        [@state.clone, reward, done]
      end


   # This method is used to render the environment
    def render(output_stream)
      @size.times do |i|
        @size.times do |j|
          if @state.include?([0, i, j, -1])
        output_stream.print 'A'
          elsif @state.include?([1, i, j, -1])
        output_stream.print 'G'
          elsif @state.include?([2, i, j, 0])
        output_stream.print 'O'
          else
        output_stream.print '.'
          end
        end
        output_stream.print "\n"
      end
    end


    end
  end
end
