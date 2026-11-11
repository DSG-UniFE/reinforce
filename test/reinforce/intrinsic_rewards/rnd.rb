# frozen_string_literal: true

require "reinforce"
require_relative "../../support/torch_test_helpers"

describe Reinforce::IntrinsicRewards::RND do
  let(:familiar) { [1.0, 0.0, 0.0, 0.0] }
  let(:novel) { [0.0, 0.0, 0.0, 1.0] }

  it "pays less for states it has been trained on than for novel ones" do
    TorchTestHelpers.with_torch_seed do
      rnd = subject.new(4)
      30.times { [familiar, [0.0, 1.0, 0.0, 0.0]].each { |state| rnd.observe(state) } }
      before = rnd.call([familiar, novel])

      20.times { rnd.train([familiar, [0.0, 1.0, 0.0, 0.0]]) }
      after = rnd.call([familiar, novel])

      expect(after[0]).to be < before[0] / 10
      expect(after[1]).to be > after[0] * 10
    end
  end

  it "trains only the predictor, keeping the random target fixed" do
    TorchTestHelpers.with_torch_seed do
      rnd = subject.new(4)
      target_snapshot = TorchTestHelpers.parameter_snapshot(rnd.target)
      predictor_snapshot = TorchTestHelpers.parameter_snapshot(rnd.predictor)

      rnd.train([familiar, novel])

      expect(TorchTestHelpers.parameters_changed?(rnd.target, target_snapshot)).to be == false
      expect(TorchTestHelpers.parameters_changed?(rnd.predictor, predictor_snapshot)).to be == true
    end
  end

  it "keeps running per-dimension observation statistics" do
    rnd = subject.new(2)
    states = [[1.0, 10.0], [3.0, 10.0], [5.0, 10.0]]
    states.each { |state| rnd.observe(state) }

    reference = states.transpose.map { |column| Reinforce::RunningMeanStd.new.tap { |stats| column.each { |value| stats.update(value) } } }
    mean = rnd.instance_variable_get(:@observation_mean).to_a
    var = rnd.instance_variable_get(:@observation_var).to_a
    expect(mean.zip(reference.map(&:mean)).all? { |actual, wanted| (actual - wanted).abs < 1e-4 }).to be == true
    expect(var.zip(reference.map(&:var)).all? { |actual, wanted| (actual - wanted).abs < 1e-4 }).to be == true
  end

  it "clips normalized observations, scales the bonus, and accepts Tensors" do
    TorchTestHelpers.with_torch_seed do
      rnd = subject.new(2, coefficient: 2.0, observation_clip: 1.0)
      clipped = rnd.send(:normalize, Torch.tensor([[1000.0, -1000.0]])).to_a.first
      unscaled = subject.new(2, rnd.config, coefficient: 1.0)
      unscaled.predictor.load_state_dict(rnd.predictor.state_dict)
      unscaled.target.load_state_dict(rnd.target.state_dict)

      expect(clipped).to be == [1.0, -1.0]
      expect(rnd.call([Torch.tensor([0.5, 0.5])]).first).to be_within(1e-6).of(2 * unscaled.call([[0.5, 0.5]]).first)
      expect(rnd.call([])).to be == []
    end
  end

  it "plugs into PPO as an intrinsic reward" do
    TorchTestHelpers.with_torch_seed do
      environment = Reinforce::Environments::GridWorld.four_rooms(size: 5, observation: :one_hot)
      rnd = subject.new(environment.state_size)
      ppo = Reinforce::Algorithms::PPO.new(environment, intrinsic_reward: rnd, ppo_epochs: 1)

      ppo.train(episodes: 2, steps_per_episode: 20)

      expect(ppo.metrics[:intrinsic_reward].size).to be > 0
      expect(ppo.metrics[:intrinsic_reward].all? { |bonus| bonus.finite? && bonus >= 0 }).to be == true
    end
  end
end
