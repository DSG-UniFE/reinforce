# frozen_string_literal: true

require 'reinforce/environments/taxi'

describe Reinforce::Environments::Taxi do
  let(:environment) do
    srand(1234)
    Reinforce::Environments::Taxi.new
  end

  it 'exposes a stable environment contract' do
    state = environment.reset
    expect(state.size).to be == 2
    expect(environment.state_size).to be == 2
    expect(environment.actions.size).to be == 6
  end

  it 'returns [state, reward, done] with both symbolic and indexed actions' do
    next_state_a, reward_a, done_a = environment.step(:south)
    next_state_b, reward_b, done_b = environment.step(0)

    expect(next_state_a.size).to be == 2
    expect(next_state_b.size).to be == 2
    expect(reward_a.is_a?(Numeric)).to be == true
    expect(reward_b.is_a?(Numeric)).to be == true
    expect([true, false].include?(done_a)).to be == true
    expect([true, false].include?(done_b)).to be == true
  end
end
