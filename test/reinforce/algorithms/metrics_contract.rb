# frozen_string_literal: true

require "reinforce"
require_relative "../../support/torch_test_helpers"

# Every online algorithm records into a Reinforce::Metrics::Recorder under
# the same common names, so generic tooling (console summaries, JSON-lines
# files, benchmarks) works on any of them without per-algorithm special
# cases. `#logs` remains as an alias of `#metrics` for existing callers.
describe "online algorithm metrics contract" do
  def gridworld
    Reinforce::Environments::GridWorld.new(3, [0, 0], [2, 2], 0)
  end

  def small_network(input_size, output_size)
    Torch::NN::Sequential.new(
      Torch::NN::Linear.new(input_size, 16),
      Torch::NN::ReLU.new,
      Torch::NN::Linear.new(16, output_size)
    )
  end

  def build_agent(name)
    environment = gridworld
    state_size = environment.state_size
    num_actions = environment.actions.size

    case name
    when :sarsa
      q_function = Reinforce::QFunctionANN.new(state_size, num_actions, 0.01, 0.9)
      Reinforce::Algorithms::SARSA.new(environment, q_function, 0.3)
    when :dqn
      agent = Reinforce::Algorithms::DQN.new(environment)
      agent.instance_variable_set(:@training_start, 0)
      agent
    when :ppo
      Reinforce::Algorithms::PPO.new(environment, 0.001, nil, nil, 0.2, 1, 4, 0.95)
    when :sac
      Reinforce::Algorithms::SAC.new(
        environment,
        policy_model: small_network(state_size, num_actions),
        q1_model: small_network(state_size, num_actions),
        q2_model: small_network(state_size, num_actions),
        q1_target_model: small_network(state_size, num_actions),
        q2_target_model: small_network(state_size, num_actions),
        batch_size: 4,
        replay_size: 32,
        warmup_steps: 4
      )
    when :monte_carlo_policy_gradient
      policy = Torch::NN::Linear.new(state_size, num_actions)
      optimizer = Torch::Optim::SGD.new(policy.parameters, lr: 0.01)
      Reinforce::Algorithms::MonteCarloPolicyGradient.new(environment, policy, optimizer)
    end
  end

  [:sarsa, :dqn, :ppo, :sac, :monte_carlo_policy_gradient].each do |name|
    it "#{name} records the common metrics and forwards them to attached sinks" do
      TorchTestHelpers.with_torch_seed do
        agent = build_agent(name)
        received = Hash.new(0)
        agent.metrics.add_sink(->(key, _value, _step) { received[key] += 1 })

        agent.train(episodes: 8, steps_per_episode: 8)

        expect(agent.metrics).to be_a(Reinforce::Metrics::Recorder)
        expect(agent.logs).to be_equal(agent.metrics)
        [:episode_reward, :episode_length, :loss].each do |key|
          expect(agent.metrics[key]).not.to be(:empty?)
          expect(received[key]).to be == agent.metrics[key].size
        end
      end
    end
  end
end
