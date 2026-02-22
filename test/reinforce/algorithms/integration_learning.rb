# frozen_string_literal: true

require 'reinforce/algorithms/dqn'
require 'reinforce/algorithms/ppo'
require 'reinforce/q_function_ann'

class BinaryChoiceEnv
  def state_size
    1
  end

  def actions
    [:bad, :good]
  end

  def reset
    [0.0]
  end

  def step(action)
    action_index = action.is_a?(Integer) ? action : actions.index(action)
    reward = action_index == 1 ? 2.0 : 1.0
    [[0.0], reward, true]
  end
end

describe 'integration learning smoke tests' do
  it 'DQN learns to value the better action on a deterministic one-step task' do
    srand(1234)
    Torch.manual_seed(1234)
    env = BinaryChoiceEnv.new

    q_layer = Torch::NN::Linear.new(1, 2)
    q_target_layer = Torch::NN::Linear.new(1, 2)
    q_arch = Torch::NN::Sequential.new(q_layer)
    q_target_arch = Torch::NN::Sequential.new(q_target_layer)
    [q_layer, q_target_layer].each do |layer|
      Torch::NN::Init.constant!(layer.weight, 0.0)
      Torch::NN::Init.constant!(layer.bias, 0.0)
    end

    q_model = Reinforce::QFunctionANN.new(1, 2, 0.05, 0.0, architecture: q_arch)
    q_target_model = Reinforce::QFunctionANN.new(1, 2, 0.05, 0.0, architecture: q_target_arch)

    agent = Reinforce::Algorithms::DQN.new(
      env,
      0.05,
      0.0,
      1.0,
      q_function_model: q_model,
      q_function_model_target: q_target_model
    )

    agent.instance_variable_set(:@training_start, 8)
    agent.instance_variable_set(:@update_frequency_for_q, 1)
    agent.instance_variable_set(:@update_frequency_for_q_target, 10)

    before_logits = q_model.forward([0.0]).to_a
    before_gap = before_logits[1] - before_logits[0]

    agent.train(200, 4)

    after_logits = q_model.forward([0.0]).to_a
    after_gap = after_logits[1] - after_logits[0]

    expect(before_gap).to be_within(1e-6).of(0.0)
    expect(after_gap > 0.0).to be == true
  end

  it 'PPO policy shifts toward the better action on a deterministic one-step task' do
    env = BinaryChoiceEnv.new

    policy_layer = Torch::NN::Linear.new(1, 2)
    value_layer = Torch::NN::Linear.new(1, 1)
    policy = Torch::NN::Sequential.new(policy_layer)
    value = Torch::NN::Sequential.new(value_layer)
    Torch::NN::Init.constant!(policy_layer.weight, 0.0)
    Torch::NN::Init.constant!(policy_layer.bias, 0.0)
    Torch::NN::Init.constant!(value_layer.weight, 0.0)
    Torch::NN::Init.constant!(value_layer.bias, 0.0)

    agent = Reinforce::Algorithms::PPO.new(env, 0.02, policy, value, 0.2, 2, 4, 0.0)

    state = Torch.tensor([0.0], dtype: :float32)
    before_logits = agent.agent.policy_model.call(state).to_a
    before_gap = before_logits[1] - before_logits[0]

    agent.train(80, 8)

    after_logits = agent.agent.policy_model.call(state).to_a
    after_gap = after_logits[1] - after_logits[0]

    expect(before_gap).to be_within(1e-6).of(0.0)
    expect(after_gap > 0.1).to be == true
  end
end
