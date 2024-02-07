module Reinforce
  module Environments

    class TaxiV2 
      attr_reader :state, :reward, :done
      def initialize
        @action = 0
        @reward = 0
        @done = false
        @num_location = 5
        @num_passenger = 1
        @num_destination = 1
        reset
      end

      def state_size
        @state.size
      end

      def reset 
        @done = false
        taxi_location = Array.new(2) { rand(@num_location) }
        passenger_location = Array.new(2) { rand(@num_location) }
        destination = Array.new(2) { rand(@num_location) }
        @passenger_in_taxi = 0
        @state = [taxi_location.dup, passenger_location.dup, destination.dup].flatten
      end

    
      def actions
        [:south, :north, :west, :east, :pickup, :dropoff]
      end

      def step(action)
        action = actions[action] if action.is_a?(Integer)
        reward = 0
        taxi_location = @state.dup[0..1]
        passenger_location = @state.dup[2..3]
        destination = @state.dup[4..5]
        
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
          if taxi_location == passenger_location && @passenger_in_taxi == 0
            @passenger_in_taxi = 1
            reward = 2
          else
            reward = -5
          end
        when :dropoff
          if taxi_location == destination && @passenger_in_taxi == 1
            reward = 2
            @done = true
            warn "Task Completed!"
          else
            reward = -5
          end
        end

        # if the agent does not pick up the passenger, it will be penalized
        reward += -5 if taxi_location == passenger_location && @passenger_in_taxi.zero? && action != :pickup
        # if the agent does not drop off the passenger, it will be penalized
        reward += -5 if taxi_location == destination && @passenger_in_taxi == 1 && action != :dropoff

        if @passenger_in_taxi == 0
          distance_to_passengers_after = (taxi_location[0] - passenger_location[0]).abs + (taxi_location[1] - passenger_location[1]).abs / @num_location.to_f
          distance_to_passengers_before = (state[0] - passenger_location[0]).abs + (state[1] - passenger_location[1]).abs / @num_location.to_f
          reward += distance_to_passengers_after < distance_to_passengers_before ? 1 : -1
        else 
          distance_to_goal_after = (taxi_location[0] - destination[0]).abs + (taxi_location[1] - destination[1]).abs / @num_location.to_f
          distance_to_goal_before = (state[0] - destination[0]).abs + (state[1] - destination[1]).abs / @num_location.to_f
          reward += distance_to_goal_after < distance_to_goal_before ? 1 : -1
        end

        @state = [taxi_location.dup, passenger_location.dup, destination.dup].flatten

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
          elsif i == @state[2] && j == @state[3] && @passenger_in_taxi == 0
            print 'P'
          elsif i == @state[4] && j == @state[5]
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
