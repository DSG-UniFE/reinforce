# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi.

require 'reinforce/algorithms/reinforce'

describe Reinforce::Algorithms::Reinforce do
  let(:num_states) { 4 }
  let(:num_actions) { 2 }
  let(:discount_factor) { 0.99 }
  let(:model) { Object.new }
  let(:optimizer) { Object.new }
  let(:agent) { Reinforce::Algorithms::Reinforce.new(num_states, num_actions, discount_factor, model, optimizer) }

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
