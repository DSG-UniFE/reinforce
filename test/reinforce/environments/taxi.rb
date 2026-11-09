# frozen_string_literal: true

require "reinforce/environments/taxi"
require_relative "../../support/environment_contract"

describe Reinforce::Environments::Taxi do
  # `Taxi` is parametrized to cover what used to be two separate,
  # hand-duplicated classes -- see lib/reinforce/environments/taxi.rb.

  describe "default configuration (randomized start, taxi-only observation)" do
    include_context EnvironmentContract, factory: -> {
      srand(1234)
      Reinforce::Environments::Taxi.new
    }

    let(:environment) do
      srand(1234)
      Reinforce::Environments::Taxi.new
    end

    it "exposes a stable environment contract" do
      state = environment.reset
      expect(state.size).to be == 2
      expect(environment.state_size).to be == 2
      expect(environment.actions.size).to be == 6
    end

    it "returns [state, reward, done] with both symbolic and indexed actions" do
      next_state_a, reward_a, done_a = environment.step(:south)
      next_state_b, reward_b, done_b = environment.step(0)

      expect(next_state_a.size).to be == 2
      expect(next_state_b.size).to be == 2
      expect(reward_a.is_a?(Numeric)).to be == true
      expect(reward_b.is_a?(Numeric)).to be == true
      expect([true, false].include?(done_a)).to be == true
      expect([true, false].include?(done_b)).to be == true
    end

    it "rewards a successful dropoff with the default dropoff_reward of 1" do
      environment.reset
      environment.instance_variable_set(:@taxi_location, environment.instance_variable_get(:@passenger_location))
      environment.step(:pickup)
      environment.instance_variable_set(:@taxi_location, environment.instance_variable_get(:@destination))
      _next_state, reward, done = environment.step(:dropoff)

      expect(reward).to be == 1
      expect(done).to be == true
    end

    it "reports pickup and dropoff in the step info instead of printing them" do
      environment.reset
      environment.instance_variable_set(:@taxi_location, environment.instance_variable_get(:@passenger_location))
      _, _, _, pickup_info = environment.step(:pickup)
      environment.instance_variable_set(:@taxi_location, environment.instance_variable_get(:@destination))
      _, _, _, dropoff_info = environment.step(:dropoff)

      expect(pickup_info).to be == {event: :passenger_picked_up}
      expect(dropoff_info).to be == {event: :task_completed}
    end

    it "returns an empty step info for ordinary moves" do
      environment.reset
      _, _, _, info = environment.step(:south)

      expect(info).to be == {}
    end
  end

  describe "full-observation fixed-start configuration (formerly TaxiV2)" do
    include_context EnvironmentContract, factory: -> {
      Reinforce::Environments::Taxi.new(randomize: false, observe_passenger_and_destination: true, dropoff_reward: 20)
    }

    let(:environment) do
      Reinforce::Environments::Taxi.new(randomize: false, observe_passenger_and_destination: true, dropoff_reward: 20)
    end

    it "exposes a stable environment contract" do
      state = environment.reset
      expect(state.size).to be == 7
      expect(environment.state_size).to be == 7
      expect(environment.actions.size).to be == 6
    end

    it "returns [state, reward, done] with both symbolic and indexed actions" do
      next_state_a, reward_a, done_a = environment.step(:east)
      next_state_b, reward_b, done_b = environment.step(3)

      expect(next_state_a.size).to be == 7
      expect(next_state_b.size).to be == 7
      expect(reward_a.is_a?(Numeric)).to be == true
      expect(reward_b.is_a?(Numeric)).to be == true
      expect([true, false].include?(done_a)).to be == true
      expect([true, false].include?(done_b)).to be == true
    end

    it "always resets to the same fixed taxi, passenger, and destination positions" do
      state_a = environment.reset
      environment.step(:east)
      state_b = environment.reset

      expect(state_a).to be == state_b
      expect(state_a).to be == [0.0, 0.0, 2.0, 2.0, 3.0, 3.0, 0.0]
    end

    it "rewards a successful dropoff with the configured dropoff_reward of 20" do
      # Fixed positions (see the reset test above): taxi [0, 0], passenger
      # [2, 2], destination [3, 3] -- navigate there directly instead of
      # reaching into the environment's internals.
      environment.reset
      environment.step(:east)
      environment.step(:east)
      environment.step(:south)
      environment.step(:south) # taxi now at [2, 2], the passenger's location
      environment.step(:pickup)
      environment.step(:east)
      environment.step(:south) # taxi now at [3, 3], the destination
      _next_state, reward, done = environment.step(:dropoff)

      expect(reward).to be == 20
      expect(done).to be == true
    end

    it "observes whether the passenger is in the taxi" do
      # Regression: the passenger's coordinates don't follow the taxi, so
      # without this flag the states before and after pickup were identical.
      environment.reset
      environment.step(:east)
      environment.step(:east)
      environment.step(:south)
      before_pickup = environment.step(:south)[0] # taxi at the passenger, [2, 2]
      after_pickup = environment.step(:pickup)[0]

      expect(before_pickup.last).to be_within(1e-9).of(0.0)
      expect(after_pickup.last).to be_within(1e-9).of(1.0)
      expect(after_pickup.first(6)).to be == before_pickup.first(6)
      expect(environment.reset.last).to be_within(1e-9).of(0.0)
    end

    it "penalizes a wrong pickup or dropoff with -10 by default" do
      environment.reset # taxi at [0, 0]: no passenger here, none on board

      expect(environment.step(:pickup)[1]).to be == -10
      expect(environment.step(:dropoff)[1]).to be == -10
    end

    it "penalizes a wrong pickup or dropoff with the configured illegal_action_reward" do
      mild = Reinforce::Environments::Taxi.new(randomize: false, observe_passenger_and_destination: true, illegal_action_reward: -2)

      expect(mild.step(:pickup)[1]).to be == -2
      expect(mild.step(:dropoff)[1]).to be == -2
      expect(mild.step(:east)[1]).to be == -1
    end
  end
end
