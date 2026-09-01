# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi.

require "reinforce/algorithms/sarsa"
require "torch"

describe Reinforce::Algorithms::SARSA do
  let(:environment) { Object.new }
  let(:q_function_model) { Object.new }
  let(:agent) { Reinforce::Algorithms::SARSA.new(environment, q_function_model) }

  it "can be instantiated" do
    expect(agent).not.to be_nil
  end

  it "can be saved" do
    expect(agent.respond_to?(:save)).to be == true
  end

  it "can be loaded" do
    expect(agent.respond_to?(:load)).to be == true
  end

  it "chooses next action from next_state during training" do
    q_model = Object.new
    q_model.instance_variable_set(:@seen_states, [])
    q_model.instance_variable_set(:@update_on_policy_values, [])
    q_model.define_singleton_method(:seen_states) { @seen_states }
    q_model.define_singleton_method(:update_on_policy_values) { @update_on_policy_values }
    q_model.define_singleton_method(:random_action) do |state|
      @seen_states << state.dup
      0
    end
    q_model.define_singleton_method(:update) do |_experience, on_policy: false|
      @update_on_policy_values << on_policy
      {loss: 0.0}
    end
    q_model.define_singleton_method(:save) { |_path| nil }
    q_model.define_singleton_method(:load) { |_path| nil }

    env = Object.new
    env.define_singleton_method(:reset) { [0] }
    env.define_singleton_method(:step) { |_action| [[1], 1.0, true] }

    sarsa = Reinforce::Algorithms::SARSA.new(env, q_model, 1.0)
    sarsa.train(episodes: 1, steps_per_episode: 2)

    expect(q_model.seen_states).to be == [[0], [1]]
  end

  it "delegates to QFunctionANN#update with on_policy: true" do
    # Regression test for the SARSA-vs-Q-learning bug: SARSA computes an
    # on-policy trajectory, so it must tell QFunctionANN#update to
    # bootstrap from the actual next_action taken, not the greedy one.
    # See lib/reinforce/q_function_ann.rb and lib/reinforce/algorithms/sarsa.rb.
    q_model = Object.new
    q_model.instance_variable_set(:@update_on_policy_values, [])
    q_model.define_singleton_method(:update_on_policy_values) { @update_on_policy_values }
    q_model.define_singleton_method(:random_action) { |_state| 0 }
    q_model.define_singleton_method(:forward) { |_state| Torch.tensor([1.0, 0.0]) }
    q_model.define_singleton_method(:update) do |_experience, on_policy: false|
      @update_on_policy_values << on_policy
      {loss: 0.0}
    end
    q_model.define_singleton_method(:save) { |_path| nil }
    q_model.define_singleton_method(:load) { |_path| nil }

    env = Object.new
    env.define_singleton_method(:reset) { [0] }
    env.define_singleton_method(:step) { |_action| [[1], 1.0, true] }

    sarsa = Reinforce::Algorithms::SARSA.new(env, q_model, 1.0)
    sarsa.train(episodes: 2, steps_per_episode: 2)

    expect(q_model.update_on_policy_values).to be == [true, true]
  end
end
