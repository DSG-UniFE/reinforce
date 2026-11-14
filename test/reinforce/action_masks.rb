# frozen_string_literal: true

require "reinforce"

# Two states, shown one-hot. Action 3 pays 1.0 but is masked in state 0,
# where the best allowed action is 2 (0.5); the others pay nothing.
# Records every action taken and whether it was allowed.
class MaskedBandit
  include Reinforce::Environment

  attr_reader :taken

  def initialize(seed: 1)
    @random = Random.new(seed)
    @taken = []
  end

  def state_size = 2

  def actions = [0, 1, 2, 3]

  def reset
    @state = @random.rand(2)
    observation
  end

  def action_mask = [true, true, true, @state == 1]

  def step(action)
    action = action.to_i
    @taken << [@state, action, action_mask[action]]
    reward = if action == 3 then 1.0
    elsif action == 2 then 0.5
    else 0.0
    end
    @state = @random.rand(2)
    [observation, reward, false, {}]
  end

  def observation = (@state == 0) ? [1.0, 0.0] : [0.0, 1.0]
end

# Two choices per step, 3 options each: the reward is 1 when they match
# the state's target pair, which in state 0 is [2, 1]; option 0 of the
# second dimension is always masked.
class MaskedPairs
  include Reinforce::Environment

  TARGETS = [[2, 1], [0, 2]].freeze

  attr_reader :taken

  def initialize(seed: 1)
    @random = Random.new(seed)
    @taken = []
  end

  def state_size = 2

  def action_space = Reinforce::Spaces::MultiDiscrete.new([3, 3])

  def reset
    @state = @random.rand(2)
    observation
  end

  def action_mask = [[true, true, true], [false, true, true]]

  def step(action)
    @taken << action
    reward = (action == TARGETS[@state]) ? 1.0 : 0.0
    @state = @random.rand(2)
    [observation, reward, false, {}]
  end

  def observation = (@state == 0) ? [1.0, 0.0] : [0.0, 1.0]
end

describe "Action masks" do
  def logits = Torch.tensor([0.0, 5.0, 1.0, 2.0])

  it "never samples a masked action, and chooses the best allowed one greedily" do
    Torch.manual_seed(1)
    distribution = CategoricalDistribution.new(logits:, mask: [true, false, true, true])
    samples = Array.new(500) { distribution.choose(deterministic: false) }

    expect(samples.include?(1)).to be == false
    expect(distribution.choose(deterministic: true)).to be == 3
    expect(distribution.entropy.item.finite?).to be == true
    expect(distribution.log_probability(1).item).to be < -1e8
  end

  it "splits multi-categorical logits by dimension and sums log-probabilities" do
    logits = Torch.tensor([[0.0, 1.0, 2.0, 3.0, 0.0], [1.0, 0.0, 0.0, 0.0, 4.0]])
    distribution = MultiCategoricalDistribution.new(logits:, sizes: [3, 2])
    actions = Torch.tensor([[2, 0], [0, 1]])
    expected = Torch.log_softmax(logits[0.., 0...3], dim: -1).to_a.zip([2, 0]).map { |row, i| row[i] }
      .zip(Torch.log_softmax(logits[0.., 3...5], dim: -1).to_a.zip([0, 1]).map { |row, i| row[i] }).map(&:sum)

    distribution.log_probability(actions).to_a.zip(expected).each { |got, want| expect(got).to be_within(1e-5).of(want) }
    expect(distribution.entropy.size.to_a).to be == [2]
    masked = MultiCategoricalDistribution.new(logits: logits[0], sizes: [3, 2], mask: [true, true, false, true, false])
    expect(masked.choose(deterministic: true)).to be == [1, 0]
  end

  it "describes multi-discrete spaces" do
    space = Reinforce::Spaces::MultiDiscrete.new([3, 2])
    expect([space.size, space.total]).to be == [2, 5]
    expect(space.contains?([2, 1])).to be == true
    expect(space.contains?([3, 0])).to be == false
    expect { Reinforce::Spaces::MultiDiscrete.new([0]) }.to raise_exception(ArgumentError)
    expect(Reinforce::Spaces.multi_discrete(MaskedPairs.new)).to be == space.class.new([3, 3])
    expect(Reinforce::Spaces.multi_discrete(MaskedBandit.new)).to be_nil
  end

  it "trains PPO without ever taking a masked action" do
    Torch.manual_seed(2)
    environment = MaskedBandit.new
    agent = Reinforce::Algorithms::PPO.new(environment, learning_rate: 0.01, ppo_epochs: 4, minibatch_size: 32, normalize_rewards: false)
    agent.train(episodes: 40, steps_per_episode: 64)

    expect(environment.taken.all?(&:last)).to be == true
    expect(environment.taken.count { |_, a, _| a == 3 }).to be > 0 # state 1 still uses it
    expect(agent.predict([1.0, 0.0], mask: [true, true, true, false])).to be == 2
    expect(agent.predict([0.0, 1.0], mask: [true, true, true, true])).to be == 3
  end

  it "trains PPO on a multi-discrete space under masks" do
    Torch.manual_seed(3)
    environment = MaskedPairs.new
    agent = Reinforce::Algorithms::PPO.new(environment, learning_rate: 0.01, ppo_epochs: 4, minibatch_size: 32, normalize_rewards: false)
    agent.train(episodes: 60, steps_per_episode: 64)

    expect(environment.taken.none? { |pair| pair[1].zero? }).to be == true
    expect(agent.predict([1.0, 0.0])).to be == [2, 1]
    expect(agent.predict([0.0, 1.0])).to be == [0, 2]
  end

  it "trains DQN without ever taking a masked action, bootstrapping over allowed ones" do
    Torch.manual_seed(4)
    srand(4)
    environment = MaskedBandit.new
    agent = Reinforce::Algorithms::DQN.new(environment, learning_rate: 0.005, training_start: 64, q_update_frequency: 4,
      target_update_frequency: 64, minibatch_size: 32, epsilon: 0.5)
    agent.train(episodes: 30, steps_per_episode: 64)

    expect(environment.taken.all?(&:last)).to be == true
    expect(agent.predict([1.0, 0.0], mask: [true, true, true, false])).to be == 2
    expect(agent.predict([0.0, 1.0], mask: [true, true, true, true])).to be == 3
  end

  it "refuses a mask that allows nothing" do
    environment = MaskedBandit.new
    environment.define_singleton_method(:action_mask) { [false, false, false, false] }
    expect { Reinforce::Spaces.flat_action_mask(environment) }.to raise_exception(ArgumentError)
  end
end
