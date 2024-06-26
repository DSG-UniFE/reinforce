module Reinforce
  module Environments

    class TaxiV2 
      attr_reader :state, :reward, :done
      def initialize
        @action = 0
        @reward = 0
        @done = false
        @num_location = 4
        @num_passenger = 1
        @num_destination = 1
        reset
      end

      def state_size
        @state.size
      end

      def reset 
        @done = false
        taxi_location = [0, 0]#Array.new(2) { rand(@num_location) }
        passenger_location = [2,2]#Array.new(2) { rand(@num_location) }
        destination = [3,3]#Array.new(2) { rand(@num_location) }
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
        reward = -1 # default reward according to gynnasyium
        
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
            warn "Passenger picked up!"
            #reward = 5
          else
            reward += -10
          end
        when :dropoff
          if taxi_location == destination && @passenger_in_taxi == 1
            reward = 20
            @done = true
            warn "Task Completed!"
          else
            reward += -10
          end
        end
# the gynnasium environment does not support that
=begin
        # if not pickup or dropoff add a reward component to drive the taxi
        # close to the pickup point (when taxi is free) or to the dropoff point
        if action != :pickup && action != :dropoff
          whereto = @passenger_in_taxi == 0 ?  @state[2..3] : @state[4..5]
          # Calculate euclidean distance between taxi_location and whereto
          reward += 1.0 / (1 + Math.sqrt((taxi_location[0] - whereto[0])**2 + (taxi_location[1] - whereto[1])**2))
        end
=end
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
