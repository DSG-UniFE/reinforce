# frozen_string_literal: true

module Reinforce
  module Environments
    ##
    # A grid-world "taxi" task: navigate to a passenger, pick them up, then
    # navigate to a destination and drop them off. Parametrized to cover
    # what used to be two separate, hand-duplicated classes (`Taxi` and
    # `TaxiV2`):
    #
    # - `randomize:` (default `true`) places the taxi, passenger, and
    #   destination at random grid cells on every #reset, matching the
    #   original `Taxi`. `randomize: false` fixes them at [0, 0], [2, 2],
    #   and [3, 3] respectively, matching the original `TaxiV2` -- useful
    #   for debugging a single scenario without exploration noise from a
    #   moving goal.
    # - `observe_passenger_and_destination:` (default `false`) controls
    #   whether the passenger's and destination's coordinates are part of
    #   the observed state, matching the original `Taxi` (2-dimensional
    #   state, taxi position only). `true` concatenates them onto the
    #   state, followed by a passenger-in-taxi flag (1.0 once picked up,
    #   0.0 before): 7-dimensional. The original `TaxiV2` had the six
    #   coordinates but no flag, which left the task partially observable
    #   -- the passenger's coordinates don't follow the taxi after pickup,
    #   so "heading to the passenger" and "carrying the passenger" were
    #   indistinguishable, and no deterministic policy could solve it.
    #   Note the
    #   `false` default means the agent cannot observe where the
    #   passenger or destination are -- that is inherited unchanged from
    #   the original `Taxi` behavior, not a recommendation; environments
    #   built with the default make pickup/dropoff unobservable from
    #   state alone.
    # - `dropoff_reward:` (default `1`, matching the original `Taxi`) is
    #   the reward for a successful dropoff; the original `TaxiV2` used
    #   `20`.
    # - `illegal_action_reward:` (default `-10`, matching the OpenAI Gym Taxi
    #   task) is the reward for a pickup or dropoff attempted in the wrong
    #   place or state. Large penalties here can teach an agent to never
    #   attempt pickup/dropoff at all (every step then costs only -1) long
    #   before it discovers the dropoff reward, so a milder value can make
    #   the task learnable.
    class Taxi
      include ::Reinforce::Environment

      attr_reader :state, :reward, :done

      def initialize(grid_size: 4, randomize: true, observe_passenger_and_destination: false, dropoff_reward: 1, illegal_action_reward: -10)
        @grid_size = grid_size
        @randomize = randomize
        @observe_passenger_and_destination = observe_passenger_and_destination
        @dropoff_reward = dropoff_reward
        @illegal_action_reward = illegal_action_reward
        @reward = 0
        @done = false
        reset
      end

      def state_size
        @state.size
      end

      def reset
        @done = false
        @taxi_location = start_location([0, 0])
        @passenger_location = start_location([2, 2])
        @destination = start_location([3, 3])
        @passenger_in_taxi = false
        @state = build_state
      end

      def actions
        [:south, :north, :west, :east, :pickup, :dropoff]
      end

      def step(action)
        action = actions[action] if action.is_a?(Integer)
        reward = -1 # default reward according to the OpenAI Gym Taxi task
        info = {}

        case action
        when :south
          @taxi_location = [@taxi_location[0], [@taxi_location[1] + 1, @grid_size - 1].min]
        when :north
          @taxi_location = [@taxi_location[0], [@taxi_location[1] - 1, 0].max]
        when :west
          @taxi_location = [[@taxi_location[0] - 1, 0].max, @taxi_location[1]]
        when :east
          @taxi_location = [[@taxi_location[0] + 1, @grid_size - 1].min, @taxi_location[1]]
        when :pickup
          if @taxi_location == @passenger_location && !@passenger_in_taxi
            @passenger_in_taxi = true
            info[:event] = :passenger_picked_up
          else
            reward = @illegal_action_reward
          end
        when :dropoff
          if @taxi_location == @destination && @passenger_in_taxi
            reward = @dropoff_reward
            @done = true
            info[:event] = :task_completed
          else
            reward = @illegal_action_reward
          end
        end

        @state = build_state

        [@state, reward, @done, info]
      end

      # Let's render the environment on the screen: a grid with the
      # position of the taxi, the passenger, and the destination.
      def render(output_stream = $stdout)
        output_stream.puts "State: #{@state}"
        (0...@grid_size).each do |j|
          line = ""
          (0...@grid_size).each do |i|
            line += if i == @taxi_location[0] && j == @taxi_location[1]
              @passenger_in_taxi ? "C" : "T"
            elsif i == @passenger_location[0] && j == @passenger_location[1] && !@passenger_in_taxi
              "P"
            elsif i == @destination[0] && j == @destination[1]
              "D"
            else
              "-"
            end
          end
          output_stream.puts line
        end
        output_stream.puts ""
      end

      private

      def start_location(fixed_location)
        @randomize ? Array.new(2) { rand(@grid_size) } : fixed_location
      end

      def build_state
        return @taxi_location.map(&:to_f) unless @observe_passenger_and_destination

        [*@taxi_location, *@passenger_location, *@destination, @passenger_in_taxi ? 1 : 0].map(&:to_f)
      end
    end
  end
end
