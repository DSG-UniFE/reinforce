# frozen_string_literal: true

require 'reinforce/environments/taxiv2'

describe Reinforce::Environments::TaxiV2 do
  let(:environment) { Reinforce::Environments::TaxiV2.new }

  it 'exposes a stable environment contract' do
    state = environment.reset
    expect(state.size).to be == 6
    expect(environment.state_size).to be == 6
    expect(environment.actions.size).to be == 6
  end

  it 'returns [state, reward, done] with both symbolic and indexed actions' do
    next_state_a, reward_a, done_a = environment.step(:east)
    next_state_b, reward_b, done_b = environment.step(3)

    expect(next_state_a.size).to be == 6
    expect(next_state_b.size).to be == 6
    expect(reward_a.is_a?(Numeric)).to be == true
    expect(reward_b.is_a?(Numeric)).to be == true
    expect([true, false].include?(done_a)).to be == true
    expect([true, false].include?(done_b)).to be == true
  end
end
