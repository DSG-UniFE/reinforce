# frozen_string_literal: true

require 'reinforce'
require 'reinforce/algorithms/sarsa'
require 'reinforce/algorithms/ppo'
require 'reinforce/q_function_ann'

describe 'trainer integration on real environments' do
  it 'trains SARSA for a short run on OneDimensionalWalk and records logs' do
    srand(1234)
    Torch.manual_seed(1234)
    environment = Reinforce::Environments::OneDimensionalWalk.new(8)
    q_function = Reinforce::QFunctionANN.new(environment.state_size, environment.actions.size, 0.01, 0.9)
    agent = Reinforce::Algorithms::SARSA.new(environment, q_function, 0.3)

    agent.train(4, 8)

    expect(agent.logs[:loss].size).to be == 4
    expect(agent.logs[:episode_reward].size > 0).to be == true
    expect(agent.logs[:episode_length].size > 0).to be == true
  end

  it 'trains PPO for a short run on GridWorld and records logs' do
    srand(1234)
    Torch.manual_seed(1234)
    environment = Reinforce::Environments::GridWorld.new(5, [0, 0], [4, 4], 0)
    agent = Reinforce::Algorithms::PPO.new(environment, 0.001, nil, nil, 0.2, 1, 4, 0.95)

    agent.train(3, 8)

    expect(agent.logs[:loss].size > 0).to be == true
    expect(agent.logs[:episode_reward].size > 0).to be == true
    expect(agent.logs[:episode_length].size > 0).to be == true
  end
end
