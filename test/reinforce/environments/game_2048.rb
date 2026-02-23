# frozen_string_literal: true

require 'reinforce/environments/game_2048'

describe Reinforce::Environments::Game2048 do
  let(:environment) do
    srand(1234)
    Reinforce::Environments::Game2048.new(4)
  end

  it 'exposes a stable environment contract' do
    state = environment.reset
    expect(state.size).to be == 16
    expect(environment.state_size).to be == 16
    expect(environment.actions).to be == %i[up down left right]
  end

  it 'returns [state, reward, done] with both symbolic and indexed actions' do
    next_state_a, reward_a, done_a = environment.step(:left)
    next_state_b, reward_b, done_b = environment.step(2)

    expect(next_state_a.size).to be == 16
    expect(next_state_b.size).to be == 16
    expect(reward_a.is_a?(Numeric)).to be == true
    expect(reward_b.is_a?(Numeric)).to be == true
    expect([true, false].include?(done_a)).to be == true
    expect([true, false].include?(done_b)).to be == true
  end

  it 'tracks score monotonically for positive reward merges' do
    environment.reset
    initial_score = environment.score
    10.times { environment.step(environment.actions.sample) }
    expect(environment.score >= initial_score).to be == true
  end
end
