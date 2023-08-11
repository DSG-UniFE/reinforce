# frozen_string_literal: true

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
end
