# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi.

require 'reinforce/experience'

describe Reinforce::Experience do
  let(:experience) { Reinforce::Experience.new }

  it 'can be instantiated' do
    expect(experience).not.to be_nil
  end

  it 'can record learning history' do
    expect(experience.history_size).to be == 0
    experience.update([0, 0], 1, [0, 1], 10, false)
    expect(experience.history_size).to be == 1
  end

  it 'can be reset' do
    experience.update([0, 0], 1, [0, 1], 10, false)
    expect(experience.history_size).to be == 1
    experience.reset
    expect(experience.history_size).to be == 0
  end
end
