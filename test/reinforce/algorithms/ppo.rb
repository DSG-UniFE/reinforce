# frozen_string_literal: true

require "reinforce/algorithms/ppo"
require_relative "../../support/agent_contract"
require_relative "../../support/torch_test_helpers"

describe Reinforce::Algorithms::PPO do
  let(:environment) do
    env = Object.new
    env.define_singleton_method(:state_size) { 2 }
    env.define_singleton_method(:actions) { [:left, :right] }
    env
  end

  let(:agent) { Reinforce::Algorithms::PPO.new(environment, learning_rate: 0.001, clip_param: 0.2, ppo_epochs: 1, minibatch_size: 2, discount_factor: 1.0) }

  include_context AgentContract, factory: -> { agent }, state: [0.1, 0.2], actions: %i[left right]

  it "builds rollout buffers with step-aligned shapes" do
    buffers = agent.build_rollout_buffers(4)

    expect(buffers[:obs].size.to_a).to be == [4, 2]
    expect(buffers[:actions].size.to_a).to be == [4]
    expect(buffers[:logprobs].size.to_a).to be == [4]
    expect(buffers[:rewards].size.to_a).to be == [4]
    expect(buffers[:dones].size.to_a).to be == [4]
    expect(buffers[:values].size.to_a).to be == [4]
  end

  it "computes gae correctly when rollout ends on terminal transition" do
    undiscounted = Reinforce::Algorithms::PPO.new(environment, agent.config, gae_lambda: 1.0)

    rewards = Torch.tensor([1.0, 1.0], dtype: :float32)
    values = Torch.tensor([0.5, 0.5], dtype: :float32)
    dones = Torch.tensor([0.0, 0.0], dtype: :float32)

    advantages, returns = undiscounted.compute_gae(rewards, values, dones, 42.0, true)

    expect(advantages.to_a).to be == [1.5, 0.5]
    expect(returns.to_a).to be == [2.0, 1.0]
  end

  it "cuts off advantage propagation across mid-rollout terminal boundaries" do
    undiscounted = Reinforce::Algorithms::PPO.new(environment, agent.config, gae_lambda: 1.0)

    rewards = Torch.tensor([1.0, 1.0, 1.0], dtype: :float32)
    values = Torch.tensor([0.0, 0.0, 0.0], dtype: :float32)
    dones = Torch.tensor([0.0, 1.0, 0.0], dtype: :float32)

    advantages, returns = undiscounted.compute_gae(rewards, values, dones, 2.0, false)

    expect(advantages.to_a).to be == [1.0, 4.0, 3.0]
    expect(returns.to_a).to be == [1.0, 4.0, 3.0]
  end

  it "splits rollout indices into non-overlapping minibatches of minibatch_size" do
    # Regression: minibatches were sliced with an inclusive range
    # (b_inds[start..start + minibatch_size]), so each had minibatch_size + 1
    # samples and consecutive minibatches overlapped by one.
    indices = [4, 0, 3, 1, 2]
    batches = agent.minibatches(indices)

    expect(batches).to be == [[4, 0], [3, 1], [2]]
    expect(batches.flatten).to be == indices
  end

  it "lets the entropy bonus affect training" do
    # Regression: #train re-wrapped the entropy tensor in Torch.tensor(...),
    # which drops its autograd history, so the entropy term contributed no
    # gradient: training with and without it produced identical weights.
    steppable = Object.new
    steppable.define_singleton_method(:state_size) { 2 }
    steppable.define_singleton_method(:actions) { %i[left right] }
    steppable.define_singleton_method(:reset) { [0.1, 0.2] }
    steppable.define_singleton_method(:step) { |action| [[0.1, 0.2], (action == 1) ? 1.0 : 0.0, false, {}] }

    policy_weights = [0.0, 0.5].map do |coefficient|
      TorchTestHelpers.with_torch_seed do
        ppo = Reinforce::Algorithms::PPO.new(steppable, learning_rate: 0.01, clip_param: 0.2, ppo_epochs: 1, minibatch_size: 4, discount_factor: 0.99, entropy_coefficient: coefficient)
        ppo.train(episodes: 2, steps_per_episode: 8)
        TorchTestHelpers.parameter_snapshot(ppo.agent.policy_model).flatten
      end
    end

    expect(policy_weights[0].zip(policy_weights[1]).any? { |a, b| (a - b).abs > 1e-6 }).to be == true
  end

  it "predicts the same action from an Array state and the equivalent Tensor state" do
    # Regression: #predict converted its input with
    # `argument = Torch.tensor(...) unless state.is_a?(Torch::Tensor)`, which
    # left argument nil -- and crashed the policy network -- whenever the
    # state was already a Tensor. #predict is greedy by default, so equal
    # actions also check both input forms reach the policy as the same values.
    array_action = agent.predict([0.1, 0.2])
    tensor_action = agent.predict(Torch.tensor([0.1, 0.2], dtype: :float32))

    expect(environment.actions).to be(:include?, array_action)
    expect(tensor_action).to be == array_action
  end

  # A one-dimensional environment whose observation is the number of steps
  # taken since the last reset, rewarding 1.0 per step and terminating after
  # `episode_length` steps (never, if nil).
  def counting_environment(episode_length: nil, observation_type: Float)
    env = Object.new
    steps = 0
    observe = ->(count) { [(observation_type == Float) ? count.to_f : count] }
    env.define_singleton_method(:state_size) { 1 }
    env.define_singleton_method(:actions) { %i[left right] }
    env.define_singleton_method(:reset) { observe.call(steps = 0) }
    env.define_singleton_method(:step) do |_action|
      steps += 1
      [observe.call(steps), 1.0, !episode_length.nil? && steps >= episode_length, {}]
    end
    env
  end

  it "trains on Integer observations without altering the environment's own arrays" do
    # Regression: only #reset observations were converted to floats (by
    # mutating the returned array in place), so Integer observations from
    # #step reached the policy as a Long tensor and crashed it.
    environment = counting_environment(episode_length: 3, observation_type: Integer)
    initial = environment.reset
    ppo = Reinforce::Algorithms::PPO.new(environment, learning_rate: 0.001, clip_param: 0.2, ppo_epochs: 1, minibatch_size: 4)

    ppo.train(episodes: 1, steps_per_episode: 6)

    expect(ppo.metrics[:episode_length]).to be == [3, 3]
    expect(initial).to be == [0]
    expect(initial.first).to be_a(Integer)
  end

  it "records each episode's own reward, not a running total across a rollout" do
    # Regression: episode_reward was only reset once per rollout, so the
    # second and later episodes in a rollout logged cumulative rewards.
    ppo = Reinforce::Algorithms::PPO.new(counting_environment(episode_length: 3), learning_rate: 0.001, clip_param: 0.2, ppo_epochs: 1, minibatch_size: 4)
    ppo.train(episodes: 1, steps_per_episode: 9)

    expect(ppo.metrics[:episode_reward]).to be == [3.0, 3.0, 3.0]
    expect(ppo.metrics[:episode_length]).to be == [3, 3, 3]
  end

  it "anneals the learning rate linearly over the rollouts" do
    # Regression: the annealing fraction used Integer division, which is
    # zero for every rollout, so the learning rate never changed.
    ppo = Reinforce::Algorithms::PPO.new(counting_environment, learning_rate: 0.01, clip_param: 0.2, ppo_epochs: 1, minibatch_size: 4)
    seen = []
    optimizer = ppo.optimizer
    optimizer.define_singleton_method(:step) do |*args|
      seen << param_groups[0][:lr]
      super(*args)
    end

    ppo.train(episodes: 4, steps_per_episode: 4)

    expect(seen.uniq.map { |lr| lr.round(6) }).to be == [0.01, 0.0075, 0.005, 0.0025]
  end

  it "bootstraps a rollout-truncated episode from the state it reached, not the reset state" do
    # Regression: the environment was reset before the bootstrap value was
    # computed, so a truncated episode bootstrapped from the start state.
    ppo = Reinforce::Algorithms::PPO.new(counting_environment, learning_rate: 0.001, clip_param: 0.2, ppo_epochs: 1, minibatch_size: 4)
    bootstrapped_from = []
    network = ppo.agent
    network.define_singleton_method(:get_value) do |x|
      bootstrapped_from << x.to_a
      super(x)
    end

    ppo.train(episodes: 1, steps_per_episode: 5)

    expect(bootstrapped_from).to be == [[5.0]]
  end

  describe "reward normalization" do
    # Records the rewards PPO stores for the update, by spying on GAE.
    def stored_rewards(ppo, episodes:, steps_per_episode:)
      seen = []
      ppo.define_singleton_method(:compute_gae) do |rewards, *rest|
        seen.concat(rewards.to_a)
        super(rewards, *rest)
      end
      ppo.train(episodes:, steps_per_episode:)
      seen
    end

    it "is on by default and divides rewards by the running std of the discounted return" do
      ppo = Reinforce::Algorithms::PPO.new(counting_environment(episode_length: 3), learning_rate: 0.001, clip_param: 0.2, ppo_epochs: 1, minibatch_size: 4, discount_factor: 0.5)
      rewards = stored_rewards(ppo, episodes: 1, steps_per_episode: 6)

      # Reference computation, with the accumulator reset after each
      # 3-step episode: discounted returns 1, 1.5, 1.75, 1, 1.5, 1.75.
      reference = Reinforce::RunningMeanStd.new
      expected = [1.0, 1.5, 1.75, 1.0, 1.5, 1.75].map do |discounted_return|
        reference.update(discounted_return)
        1.0 / Math.sqrt(reference.var + Reinforce::Algorithms::PPO::REWARD_NORMALIZATION_EPSILON)
      end
      expect(rewards.zip(expected).all? { |actual, wanted| (actual - wanted).abs < 1e-5 }).to be == true
      expect(rewards.all? { |reward| reward > 1.0 }).to be == true
    end

    it "keeps raw rewards in the episode_reward metric" do
      ppo = Reinforce::Algorithms::PPO.new(counting_environment(episode_length: 3), learning_rate: 0.001, clip_param: 0.2, ppo_epochs: 1, minibatch_size: 4)
      ppo.train(episodes: 1, steps_per_episode: 6)

      expect(ppo.metrics[:episode_reward]).to be == [3.0, 3.0]
    end

    it "stores raw rewards when disabled" do
      ppo = Reinforce::Algorithms::PPO.new(counting_environment, learning_rate: 0.001, clip_param: 0.2, ppo_epochs: 1, minibatch_size: 4, normalize_rewards: false)

      expect(stored_rewards(ppo, episodes: 1, steps_per_episode: 4)).to be == [1.0, 1.0, 1.0, 1.0]
    end
  end

  describe "intrinsic reward" do
    it "with intrinsic_value_head: false, adds the bonus to the rewards it learns from, keeping metrics extrinsic" do
      bonus = Reinforce::IntrinsicRewards::CountBased.new(coefficient: 1.0)
      ppo = Reinforce::Algorithms::PPO.new(
        counting_environment(episode_length: 2), ppo_epochs: 1, minibatch_size: 4, normalize_rewards: false, intrinsic_reward: bonus,
        intrinsic_value_head: false
      )
      learned = []
      ppo.define_singleton_method(:compute_gae) do |rewards, *rest|
        learned.concat(rewards.to_a)
        super(rewards, *rest)
      end

      ppo.train(episodes: 1, steps_per_episode: 4)

      # States 1, 2, 1, 2: the second visit to each pays 1/sqrt(2).
      expected = [2.0, 2.0, 1 + 1 / Math.sqrt(2), 1 + 1 / Math.sqrt(2)]
      expect(learned.zip(expected).all? { |actual, wanted| (actual - wanted).abs < 1e-5 }).to be == true
      expect(ppo.metrics[:episode_reward]).to be == [2.0, 2.0]
      expect(ppo.metrics[:intrinsic_reward].size).to be == 2
      expect(bonus.count([1.0])).to be == 2
    end
  end

  describe "separate intrinsic value head" do
    def two_head_agent(**options)
      Reinforce::Algorithms::PPO.new(
        counting_environment(episode_length: 2), ppo_epochs: 1, minibatch_size: 4, normalize_rewards: false,
        intrinsic_reward: Reinforce::IntrinsicRewards::CountBased.new(coefficient: 1.0), intrinsic_value_head: true,
        intrinsic_discount_factor: 0.5, **options
      )
    end

    # Records the arguments of every #compute_gae call.
    def gae_calls(ppo)
      calls = []
      ppo.define_singleton_method(:compute_gae) do |rewards, values, dones, next_value, next_done, **options|
        calls << {rewards: rewards.to_a, dones: dones.to_a, next_done:, **options}
        super(rewards, values, dones, next_value, next_done, **options)
      end
      calls
    end

    it "keeps the bonus out of the extrinsic stream and learns a non-episodic intrinsic one" do
      ppo = two_head_agent
      calls = gae_calls(ppo)

      ppo.train(episodes: 1, steps_per_episode: 4)

      extrinsic, intrinsic = calls
      expect(extrinsic[:rewards]).to be == [1.0, 1.0, 1.0, 1.0]
      expect(intrinsic[:discount_factor]).to be_within(1e-12).of(0.5)
      expect(intrinsic[:dones]).to be == [0.0, 0.0, 0.0, 0.0]
      expect(intrinsic[:next_done]).to be == false
      # Bonuses 1, 1, 1/sqrt(2), 1/sqrt(2), divided by the running std of the
      # (non-episodic, gamma 0.5) discounted intrinsic return.
      reference = Reinforce::RunningMeanStd.new
      discounted = 0.0
      expected = [1.0, 1.0, 1 / Math.sqrt(2), 1 / Math.sqrt(2)].map do |bonus|
        discounted = discounted * 0.5 + bonus
        reference.update(discounted)
        bonus / Math.sqrt(reference.var + 1e-8)
      end
      expect(intrinsic[:rewards].zip(expected).all? { |actual, wanted| (actual - wanted).abs < 1e-5 }).to be == true
      expect(ppo.metrics[:episode_reward]).to be == [2.0, 2.0]
    end

    it "trains the intrinsic value head along with the policy and extrinsic value" do
      TorchTestHelpers.with_torch_seed do
        ppo = two_head_agent
        head = ppo.instance_variable_get(:@intrinsic_value_model)
        snapshot = TorchTestHelpers.parameter_snapshot(head)

        ppo.train(episodes: 2, steps_per_episode: 4)

        expect(ppo.trainable_parameters.size).to be == 3 * ppo.agent.value_model.parameters.size
        expect(TorchTestHelpers.parameters_changed?(head, snapshot)).to be == true
      end
    end

    it "weights the two advantages, optionally decaying the intrinsic weight with training" do
      extrinsic = Torch.tensor([1.0, -1.0])
      intrinsic = Torch.tensor([2.0, 4.0])

      constant = two_head_agent.combine_advantages(extrinsic, intrinsic, 0.5)
      decaying = two_head_agent(intrinsic_advantage_decay: true).combine_advantages(extrinsic, intrinsic, 0.5)

      # Defaults: 2 * A_ext + 1 * A_int; with decay, the 1 becomes 0.5 here.
      expect(constant.to_a).to be == [4.0, 2.0]
      expect(decaying.to_a).to be == [3.0, 0.0]
    end

    it "is on by default" do
      expect(Reinforce::Algorithms::PPO::Config.new.intrinsic_value_head).to be == true
    end

    it "only exists when there is an intrinsic reward" do
      plain = Reinforce::Algorithms::PPO.new(counting_environment)

      expect(plain.instance_variable_get(:@intrinsic_value_model)).to be_nil
      expect(plain.trainable_parameters.size).to be == 2 * plain.agent.value_model.parameters.size
    end
  end
end
