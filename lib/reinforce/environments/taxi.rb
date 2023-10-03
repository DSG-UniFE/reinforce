module Reinforce
  module Environments

    class Taxi 
      attr_reader :state, :reward, :done
      def initialize
        @action = 0
        @reward = 0
        @done = false
        @num_location = 5
        @num_passenger = 1
        @num_destination = 4
        reset
      end

      def state_size
        @state.size
      end

      def reset 
        @done = false
        taxi_location = [0,0] #Array.new(2) { rand(@num_location) }
        @passenger_location = [3, 0] #Array.new(2) { rand(@num_location) }
        @destination = [4, 4] #Array.new(2) { rand(@num_location) }
        @passenger_in_taxi = 0
        @state = taxi_location.dup
      end

    
      def actions
        [:south, :north, :west, :east, :pickup, :dropoff]
      end

      def step(action)
        action = actions[action] if action.is_a?(Integer)
        reward = 0
        taxi_location = @state.dup
        
        case action
        when :south
          taxi_location = [taxi_location[0], [taxi_location[1] + 1, @num_location - 1].min]
        when :north
          taxi_location = [taxi_location[0], [taxi_location[1] - 1, 0].max]
        when :west
          taxi_location = [[taxi_location[0] - 1, 0].max, taxi_location[1]] 
        when :east
          taxi_location = [[taxi_location[0] + 1, @num_location - 1].min, taxi_location[1]]
        when :pickup
          if taxi_location == @passenger_location && @passenger_in_taxi == 0
            @passenger_in_taxi = 1
            reward = 1
          else
            reward = -1
          end
        when :dropoff
          if taxi_location == @destination && @passenger_in_taxi == 1
            reward = 1
            @done = true
          else
            reward = -1
          end
        end

        if @passenger_in_taxi == 0
          distance_to_passengers = (taxi_location[0] - @passenger_location[0]).abs + (taxi_location[1] - @passenger_location[1]).abs / @num_location.to_f
          reward -= distance_to_passengers
        else 
          distance_to_goal = (taxi_location[0] - @destination[0]).abs + (taxi_location[1] - @destination[1]).abs / @num_location.to_f
          reward -= distance_to_goal
        end

        @state = taxi_location

        [@state, reward, @done]
      end
 
    # Let's render the environment  on the screen
    # let's draw a grid with the position of the taxi, the passenger and the num_destination
    def render

      warn "State: #{@state}"
      (0...@num_location).each do |j|
        (0...@num_location).each do |i|
          if i == @state[0] && j == @state[1]
            print @passenger_in_taxi == 1 ? 'C' : 'T'
          elsif i == @passenger_location[0] && j == @passenger_location[1] && @passenger_in_taxi == 0
            print 'P'
          elsif i == @destination[0] && j == @destination[1]
            print 'D'
          else
            print '-'
          end
        end
        print "\n"
      end
      print "\n"
    end

    end
  end
end
