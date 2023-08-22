# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi.

require 'reinforce/algorithms/sarsa'

describe Reinforce::Algorithms::Sarsa do
  let(:discount_factor) { 0.99 }
  let(:q_function_model) { Object.new }
  let(:optimizer) { Object.new }
  let(:agent) { Reinforce::Algorithms::Sarsa.new(discount_factor, q_function_model, optimizer) }

  it 'can be instantiated' do
    expect(agent).not.to be_nil
  end

  it 'can record learning history' do
    expect(agent.history_size).to be == 0
    agent.update([0, 0], 1, [0, 1], 10, false)
    expect(agent.history_size).to be == 1
  end

  it 'can be reset' do
    agent.update([0, 0], 1, [0, 1], 10, false)
    expect(agent.history_size).to be == 1
    agent.reset
    expect(agent.history_size).to be == 0
  end
end
