# frozen_string_literal: true

require "reinforce/environment"

describe Reinforce::Environment do
  let(:bare_environment_class) do
    Class.new do
      include Reinforce::Environment
    end
  end

  let(:bare_environment) { bare_environment_class.new }

  it "raises NotImplementedError with a helpful message when #reset is not overridden" do
    expect do
      bare_environment.reset
    end.to raise_exception(NotImplementedError, message: be =~ /must implement #reset/)
  end

  it "raises NotImplementedError with a helpful message when #state_size is not overridden" do
    expect do
      bare_environment.state_size
    end.to raise_exception(NotImplementedError, message: be =~ /must implement #state_size/)
  end

  it "raises NotImplementedError with a helpful message when #actions is not overridden" do
    expect do
      bare_environment.actions
    end.to raise_exception(NotImplementedError, message: be =~ /must implement #actions/)
  end

  it "raises NotImplementedError with a helpful message when #step is not overridden" do
    expect do
      bare_environment.step(:some_action)
    end.to raise_exception(NotImplementedError, message: be =~ /must implement #step/)
  end

  it "defaults #render to a no-op that does not raise" do
    expect do
      bare_environment.render
    end.not.to raise_exception
  end

  it "lets a concrete class override the contract methods normally" do
    concrete_class = Class.new do
      include Reinforce::Environment

      def reset
        @state = [0]
      end

      def state_size
        1
      end

      def actions
        [:noop]
      end

      def step(_action)
        [@state, 0, true, {}]
      end
    end

    environment = concrete_class.new
    expect(environment.reset).to be == [0]
    expect(environment.state_size).to be == 1
    expect(environment.actions).to be == [:noop]
    expect(environment.step(:noop)).to be == [[0], 0, true, {}]
  end
end
