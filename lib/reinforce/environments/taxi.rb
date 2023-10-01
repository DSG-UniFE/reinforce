module Reinforce
  module Environments

    class Taxi 
      attr_reader :state, :reward, :done
      def initialize
        @action = 0
        @reward = 0
        @done = false
        @num_location = 5
        @num_passenger = 4
        @num_destination = 4
        reset
      end

      def reset 
        @done = false
        taxi_location = rand(@num_location)
        passenger_location = rand(@num_location)
        destination = rand(@num_destination)
        passenger_in_taxi = 0
        @state = [taxi_location, passenger_location, destination, passenger_in_taxi]
        #warn "Reset state: #{@state} size: #{@state.size}"
        @state.dup
      end

    
      def actions
        [:south, :north, :pickup, :dropoff]
      end

      def step(action)
        action = actions[action] if action.is_a?(Integer)
        reward = 0
        taxi_location = @state[0]
        passenger_in_taxi = @state[3]
        passenger_location = @state[1]
        destination = @state.dup[2]
        #warn "Before taxi_location: #{taxi_location} action: #{action}"
        case action
        when :south
          taxi_location = [taxi_location + 1, @num_location - 1].min
        when :north
          taxi_location = [taxi_location - 1, 0].max
        when :pickup
          if taxi_location == passenger_location && passenger_in_taxi == 0
            passenger_in_taxi = 1
            reward = 5
          else
            reward = -10
          end
        when :dropoff
          if taxi_location == destination && passenger_in_taxi == 1
            reward = 5
            @done = true
          else
            reward = -10
          end
        end

        #warn "After taxi_location: #{taxi_location} action: #{action}"
        @state = [taxi_location, passenger_location, destination, passenger_in_taxi]
        [@state, reward, done]
      end
 
      def render
        puts "Taxi location: #{@state[0]}"
        puts "Passenger location: #{@state[1]}"
        puts "Destination: #{@state[2]}}"
        puts "Passenger in taxi: #{@state[3]}"
      end
    end
  end
end
