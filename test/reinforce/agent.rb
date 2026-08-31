# frozen_string_literal: true

require "reinforce/agent"

describe Reinforce::Agent do
  let(:bare_agent_class) do
    Class.new do
      include Reinforce::Agent
    end
  end

  let(:bare_agent) { bare_agent_class.new }

  it "raises NotImplementedError with a helpful message when #predict is not overridden" do
    expect do
      bare_agent.predict([0, 0])
    end.to raise_exception(NotImplementedError, message: be =~ /must implement #predict/)
  end

  it "does not require #train, #save, or #load to be implemented" do
    # Reinforce::Agent deliberately only requires #predict: online vs. offline
    # algorithms, and algorithms like TemporalDifference that only expose
    # #learn, disagree too much on these signatures to standardize them here.
    # See lib/reinforce/agent.rb for the full rationale.
    expect(bare_agent.respond_to?(:train)).to be == false
    expect(bare_agent.respond_to?(:save)).to be == false
    expect(bare_agent.respond_to?(:load)).to be == false
  end

  it "lets a concrete class override #predict normally" do
    concrete_class = Class.new do
      include Reinforce::Agent

      def predict(state)
        state.first
      end
    end

    agent = concrete_class.new
    expect(agent.predict([:left, :right])).to be == :left
  end
end
