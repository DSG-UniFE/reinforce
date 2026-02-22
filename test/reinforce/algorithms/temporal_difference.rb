# frozen_string_literal: true

require 'reinforce/algorithms/temporal_difference'

describe Reinforce::Algorithms::TemporalDifference do
  let(:environment) do
    env = Object.new
    env.define_singleton_method(:actions) { %i[left right] }
    env
  end

  let(:agent) do
    Reinforce::Algorithms::TemporalDifference.new(
      environment,
      learning_rate: 0.5,
      discount_factor: 0.5,
      epsilon: 0.0
    )
  end

  it 'can be instantiated' do
    expect(agent).not.to be_nil
    expect(agent.q_table).to be_a(Hash)
  end

  it 'chooses the greedy action when epsilon is zero' do
    state = [0, 0]
    agent.q_table[state.freeze][:left] = 1.0
    agent.q_table[state.freeze][:right] = 2.0

    expect(agent.choose_action(state)).to be == :right
    expect(agent.predict(state)).to be == :right
  end

  it 'performs q-learning update using max next-state value' do
    state = [0, 0]
    next_state = [0, 1]
    agent.q_table[next_state.freeze][:left] = 2.0
    agent.q_table[next_state.freeze][:right] = 4.0

    # target = 1 + 0.5 * 4 = 3, update from 0 with lr=0.5 gives 1.5
    agent.learn(state, :left, 1.0, next_state)

    expect(agent.q_table[state.freeze][:left]).to be == 1.5
  end

  it 'ignores bootstrap term on terminal transitions' do
    state = [1, 1]
    next_state = [1, 2]
    agent.q_table[next_state.freeze][:left] = 100.0

    # target = reward = 2, update from 0 with lr=0.5 gives 1.0
    agent.learn(state, :right, 2.0, next_state, done: true)

    expect(agent.q_table[state.freeze][:right]).to be == 1.0
  end

  it 'supports on-policy (sarsa-style) updates with next_action' do
    state = [2, 2]
    next_state = [2, 3]
    agent.q_table[next_state.freeze][:left] = 5.0
    agent.q_table[next_state.freeze][:right] = 1.0

    # on-policy should use next_action(:right)=1, not max=5.
    # target = 1 + 0.5 * 1 = 1.5, update from 0 with lr=0.5 gives 0.75
    agent.learn(state, :left, 1.0, next_state, on_policy: true, next_action: :right)

    expect(agent.q_table[state.freeze][:left]).to be == 0.75
  end

  it 'can reset learned values' do
    state = [0, 0]
    agent.learn(state, :left, 1.0, [0, 1])
    expect(agent.q_table[state.freeze][:left]).not.to be == 0.0

    agent.reset
    expect(agent.q_table[state.freeze][:left]).to be == 0.0
  end
end
