# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi.

require 'reinforce/algorithms/sarsa'

describe Reinforce::Algorithms::SARSA do
  let(:environment) { Object.new }
  let(:q_function_model) { Object.new }
  let(:agent) { Reinforce::Algorithms::SARSA.new(environment, q_function_model) }

  it 'can be instantiated' do
    expect(agent).not.to be_nil
  end

  it 'can be saved' do
    expect(agent.respond_to?(:save)).to be == true
  end
  
  it 'can be loaded' do
    expect(agent.respond_to?(:load)).to be == true
  end

  it 'chooses next action from next_state during training' do
    q_model = Object.new
    q_model.instance_variable_set(:@seen_states, [])
    q_model.define_singleton_method(:seen_states) { @seen_states }
    q_model.define_singleton_method(:random_action) do |state|
      @seen_states << state.dup
      0
    end
    q_model.define_singleton_method(:update) { |_experience| 0.0 }
    q_model.define_singleton_method(:save) { |_path| nil }
    q_model.define_singleton_method(:load) { |_path| nil }

    env = Object.new
    env.define_singleton_method(:reset) { [0] }
    env.define_singleton_method(:step) { |_action| [[1], 1.0, true] }

    sarsa = Reinforce::Algorithms::SARSA.new(env, q_model, 1.0)
    sarsa.train(1, 2)

    expect(q_model.seen_states).to be == [[0], [1]]
  end
end
