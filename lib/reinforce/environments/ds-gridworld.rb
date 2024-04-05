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
      ACTIONS = %i[up down left right pick].freeze
      attr_reader :state
      #srand(0)
      # We want to define a state which is compatible with deep sets
      # So we need to definea state as a list of objects and the agent position
      # Each vector  will represent the item type (agent_position, goal, item) and the position
      # we use an integer to represent the item type: 0 -> agent_position, 1 -> goal, 2 -> item
      # The position is represented as a vector of two elements
      # The state is a list of vectors
      # The position of the agent is represented as following 
      # e.g [0,2,3,-1] -> agent is at position 2,3. -1 is a placeholder for dont'care
      # goal: [1, 4, 5, -1] -> goal is at position 4,5. -1 is a placeholder for dont'care
      # item: [2, 6, 7, 0] -> item is at position 6,7. here 0 means that the item is not picked up, 1 means that the item is picked up
      def initialize(size, start, goal, object_num = 5)
        @size = size
        @start = start
        @goal = goal
        @object_num = object_num
        @objects = nil 
        @state = nil
        @picked_objects = 0
        @max_steps = 250
        @step = 0
        reset
      end

      def reset
        #warn "Resetting the environment after #{@step} steps."
        #warn "state: #{@state}, picked_objects: #{@picked_objects}"
        @step = 0
        @picked_objects = 0
        @objects = Array.new(@object_num) { |_| [1 + rand(@size - 2), 1 + rand(@size - 2)] }
        encode_state
      end

      def encode_state
        @state = []
        @state << [0, @start[0], @start[1], -1].map(&:to_f)
        @state << [1, @goal[0], @goal[1], -1].map(&:to_f)
        @objects.each do |obj|
          @state << [2, obj[0], obj[1], 0].map(&:to_f)
        end 
        @state
      end

      def actions
        ACTIONS
      end

      def state_size
        [@state.size, @state[0].size]
      end

      def step(action)
        action = ACTIONS[action] if action.is_a?(Integer)
        @step += 1
        # The state is encoded so we need to decode it to get the agent position, which can be any of the vectors in the state
        # Because we are using DeepSets. Let's check the vector with the item type 0
        agent_position = @state.select { |item| item[0] == 0 }
        agent_position = agent_position[0] if agent_position != []
        # the element at position 1 and 2 are the x and y coordinates of the agent
        # There are no obstacles in this simple implementation so let's assume the agent can move freely within the grid world
        reward = 0
        done = false # The episode is not done
        case action
        when :up
          agent_position[1] -= 1 unless agent_position[1] == 0.0
        when :down
          agent_position[1] += 1 unless agent_position[1] == (@size - 1)
        when :left
          agent_position[2] -= 1 unless (agent_position[2]) == 0.0
        when :right
          agent_position[2] += 1 unless agent_position[2] == (@size - 1)
        when :pick
          # check if the agent is on an object 
          object_positions = @state.select { |item| item[0] == 2 }
          # check if the agent is on an object
          object = object_positions.select { |item| item[1] == agent_position[1] && item[2] == agent_position[2] }
          #warn "object: #{object}"
          if object == []
            # the agent selected a wrong action so we give a negative reward
            reward = -5
          else 
            # only one object can be at one position
            object = object[0]
            # the agent picked up an object, let's flag the corresponding item to 1 (picked up) 
            #warn "Picked up an object! #{object}"
            object[3] = 1.0 if object[3] == 0.0
            #warn "After pickup! #{object}"
            reward = 10
            @picked_objects += 1
          end
        else
          #raise "Invalid action: #{action}"
          # the agent selected a wrong action so we give a negative reward
          reward = -1E8 # a sort of action mapping here
        end
        # check if the agent reached a terminal state (the goal position)
        # the agent should maximize the number of objects picked up but also reach the goal
        goal_position = @state.select { |item| item[0] == 1 }
        goal_position = goal_position[0] if goal_position != []
        #warn "goal_position: #{goal_position} agent_position: #{agent_position}"
        if agent_position[1] == goal_position[1] && agent_position[2] == goal_position[2]
          #warn "Goal reached!"
          reward += 10
          done = true
        end
        # calculate the distance between the agent and the goal 
        distance_to_goal = (agent_position[1] - goal_position[1]).abs + (agent_position[2] - goal_position[2]).abs
        # maximum distance 
        max_distance = 2 * @size
        reward += -(distance_to_goal / max_distance.to_f) if reward < 1
        #warn "Returning state: #{@state}, reward: #{reward}, done: #{done}"
        return [state, reward, done]
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
