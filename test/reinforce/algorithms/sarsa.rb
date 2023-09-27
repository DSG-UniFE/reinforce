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
    expect(agent).to respond_to(:save)
  end
  
  it 'can be loaded' do
    expect(agent).to respond_to(:load)
  end
end
