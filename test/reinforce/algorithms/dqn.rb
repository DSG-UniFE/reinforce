# frozen_string_literal: true

require "reinforce/algorithms/dqn"

describe Reinforce::Algorithms::DQN do
  let(:environment) do
    env = Object.new
    env.define_singleton_method(:state_size) { 2 }
    env.define_singleton_method(:actions) { [:left, :right] }
    env
  end

  let(:agent) { Reinforce::Algorithms::DQN.new(environment, 0.001, 1.0) }

  it "can be instantiated" do
    expect(agent).not.to be_nil
  end

  it "builds a plain (non-dueling) Q-function by default" do
    expect(agent.instance_variable_get(:@q_function_model).architecture).not.to be_a(Reinforce::Models::DuelingQNetwork)
  end

  it "builds a Reinforce::Models::DuelingQNetwork-based Q-function when dueling: true" do
    dueling_agent = Reinforce::Algorithms::DQN.new(environment, dueling: true)

    expect(dueling_agent.instance_variable_get(:@q_function_model).architecture).to be_a(Reinforce::Models::DuelingQNetwork)
    expect(dueling_agent.instance_variable_get(:@q_function_model_target).architecture).to be_a(Reinforce::Models::DuelingQNetwork)
  end

  it "uses the learning_rate passed to .new for the Q-function's own optimizer" do
    # Regression test: DQN used to build its own separate optimizer with a
    # hardcoded lr: 0.001, ignoring the learning_rate constructor argument
    # entirely (it was passed into QFunctionANN.new, which built its own
    # correctly-configured optimizer, but that optimizer was never the one
    # actually used -- DQN's hand-rolled train loop stepped its own
    # shadow optimizer instead). Now that DQN delegates to
    # QFunctionANN#update, QFunctionANN's own optimizer is the only one
    # that exists, so the configured learning rate is what's actually used.
    custom_agent = Reinforce::Algorithms::DQN.new(environment, 0.5)

    lr = custom_agent.instance_variable_get(:@q_function_model).optimizer.param_groups[0][:lr]
    expect(lr).to be_within(1e-9).of(0.5)
  end

  it "passes double_dqn: through to QFunctionANN#update when training" do
    env = Object.new
    env.define_singleton_method(:state_size) { 2 }
    env.define_singleton_method(:actions) { [:left, :right] }
    env.define_singleton_method(:reset) { [0.0, 0.0] }
    env.define_singleton_method(:step) { |_action| [[0.0, 0.0], 1.0, true] }

    double_dqn_values = []
    q_model = Object.new
    q_model.define_singleton_method(:random_action) { |_state| 0 }
    q_model.define_singleton_method(:forward) { |_state| Torch.tensor([1.0, 0.0]) }
    q_model.define_singleton_method(:update) do |experience, target:, double_dqn:, weights:|
      double_dqn_values << double_dqn
      {loss: 0.0, td_errors: Array.new(experience[:indices].size, 0.0)}
    end
    q_model.define_singleton_method(:save) { |_path| nil }
    q_model.define_singleton_method(:load) { |_path| nil }

    q_target = Object.new
    q_target.define_singleton_method(:soft_update) { |_model, _tau| nil }

    dqn_agent = Reinforce::Algorithms::DQN.new(
      env, q_function_model: q_model, q_function_model_target: q_target, double_dqn: true
    )
    dqn_agent.instance_variable_set(:@training_start, 0)
    dqn_agent.instance_variable_set(:@update_frequency_for_q, 1)

    dqn_agent.train(episodes: 1, steps_per_episode: 1)

    expect(double_dqn_values).to be == [true]
  end
end
