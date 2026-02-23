# frozen_string_literal: true

require 'reinforce/environments/1d_walk'

describe Reinforce::Environments::OneDimensionalWalk do
  let(:environment) do
    srand(1234)
    Reinforce::Environments::OneDimensionalWalk.new(8)
  end

  it 'exposes a stable environment contract' do
    state = environment.reset
    expect(state.size).to be == 2
    expect(environment.state_size).to be == 2
    expect(environment.actions).to be == %i[left right]
  end

  it 'returns [state, reward, done] when stepping with symbolic and indexed actions' do
    next_state_a, reward_a, done_a = environment.step(:left)
    next_state_b, reward_b, done_b = environment.step(1)

    expect(next_state_a.size).to be == 2
    expect(next_state_b.size).to be == 2
    expect(reward_a.is_a?(Numeric)).to be == true
    expect(reward_b.is_a?(Numeric)).to be == true
    expect([true, false].include?(done_a)).to be == true
    expect([true, false].include?(done_b)).to be == true
  end

  it 'raises on invalid actions' do
    expect do
      environment.step(:invalid_action)
    end.to raise_exception(RuntimeError)
  end
end
