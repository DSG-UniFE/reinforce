# frozen_string_literal: true

require "stringio"
require "reinforce"
require "reinforce/algorithms/dqn"
require "reinforce/algorithms/ppo"
require "reinforce/algorithms/sarsa"
require "reinforce/q_function_ann"

# DQN, PPO and SARSA print a "Training: N%" progress line, but only when the
# caller asks for it via `progress_output:` -- library code should not write
# to stdout on its own (it used to, cluttering every test run).
describe "training progress output" do
  def build_agent(name)
    srand(1234)
    Torch.manual_seed(1234)
    case name
    when :dqn
      environment = Reinforce::Environments::OneDimensionalWalk.new(8)
      Reinforce::Algorithms::DQN.new(environment)
    when :ppo
      environment = Reinforce::Environments::GridWorld.new(5, [0, 0], [4, 4], 0)
      Reinforce::Algorithms::PPO.new(environment, 0.001, nil, nil, 0.2, 1, 4, 0.95)
    when :sarsa
      environment = Reinforce::Environments::OneDimensionalWalk.new(8)
      q_function = Reinforce::QFunctionANN.new(environment.state_size, environment.actions.size, 0.01, 0.9)
      Reinforce::Algorithms::SARSA.new(environment, q_function, 0.3)
    end
  end

  def capture_stdout
    original = $stdout
    $stdout = StringIO.new
    yield
    $stdout.string
  ensure
    $stdout = original
  end

  [:dqn, :ppo, :sarsa].each do |name|
    it "#{name} prints nothing by default" do
      agent = build_agent(name)
      output = capture_stdout { agent.train(episodes: 10, steps_per_episode: 4) }

      expect(output).to be == ""
    end

    it "#{name} reports progress to the given progress_output" do
      agent = build_agent(name)
      progress_output = StringIO.new
      stdout = capture_stdout { agent.train(episodes: 10, steps_per_episode: 4, progress_output:) }

      expect(progress_output.string).to be =~ /Training: \d+(\.\d+)?%/
      expect(stdout).to be == ""
    end
  end
end
