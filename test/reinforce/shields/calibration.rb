# frozen_string_literal: true

require "reinforce/shields"
require_relative "../../support/torch_test_helpers"

describe Reinforce::Shields::Calibration do
  it "scores probabilities against outcomes" do
    calibration = subject.new(bins: 2)
    calibration.add([0.9, 0.9, 0.1, 0.1], [true, false, false, false])
    calibration.add(0.6, true)

    expect(calibration.count).to be == 5
    expect(calibration.brier).to be_within(1e-9).of((0.01 + 0.81 + 0.01 + 0.01 + 0.16) / 5)
    # bin [0, 0.5): mean 0.1, frequency 0; bin [0.5, 1]: mean 0.8, frequency 2/3
    expect(calibration.expected_calibration_error).to be_within(1e-9).of((0.4 * 0.1) + (0.6 * (0.8 - (2.0 / 3))).abs)
    expect(calibration.table.map { |row| row.values_at(:low, :count) }).to be == [[0.0, 2], [0.5, 3]]
  end

  it "is perfect for well-calibrated probabilities" do
    random = Random.new(1)
    calibration = subject.new
    20_000.times do
      probability = random.rand
      calibration.add(probability, random.rand < probability)
    end

    expect(calibration.expected_calibration_error).to be < 0.02
    expect(calibration.brier).to be_within(0.01).of(1.0 / 6)
  end

  it "has nothing to say without predictions" do
    calibration = subject.new

    expect(calibration.brier).to be_nil
    expect(calibration.expected_calibration_error).to be_nil
    expect(calibration.table).to be == []
  end
end

describe Reinforce::Shields::MultiClusterSafety do
  with "telemetry" do
    let(:environment) do
      telemetry = Reinforce::Environments::MultiCluster::Telemetry.new(interval: 15.0)
      Reinforce::Environments::MultiCluster.new(clusters: 2, telemetry: telemetry,
        workload: Reinforce::Environments::MultiCluster::Workloads::Scripted.new)
    end
    let(:safety) { subject.new(environment) }

    # An observation with the given assumed free CPU of cluster 0, its age
    # (s), the others' CPU drift (per s), volatility (per sqrt(s)) and
    # probability of changing nothing over the age scale, and a request for
    # `replicas` replicas of 1 CPU (nothing else).
    def observation(free:, age:, drift:, volatility:, replicas:, quiet: 0.0)
      srand(1)
      values = environment.reset.dup
      layout = environment.observation_layout
      scale = environment.config.telemetry.age_scale
      values[layout.free] = free / 50.0
      values[layout.age] = age / scale
      values[layout.drift] = ((drift * scale / 50.0) + 1.0) / 2.0
      values[layout.volatility] = volatility * Math.sqrt(scale) / 50.0
      values[layout.quiet] = quiet
      values[layout.replicas] = replicas.to_f / environment.config.replica_limit
      demand = environment.config.max_demand_per_resource
      values[layout.demand, 4] = [1.0 / demand[:cpu], 0.0, 0.0, 0.0]
      Torch.tensor([values], dtype: :float32)
    end

    def cpu_fit(observation, share)
      safety.sensors(observation, Torch.tensor([[share, 0.0, 1.0 - share]]))[:fit][0, 0, 0].item
    end

    def normal_tail(x) = 0.5 * Math.erfc(x / Math.sqrt(2.0))

    it "takes the free amount as normal around the assumed one minus the drift, spreading with the age" do
      # 4 replicas: 4 CPUs needed; mean 10 - 0.1 * 30 = 7, sd 0.5 * sqrt(30)
      value = cpu_fit(observation(free: 10.0, age: 30.0, drift: 0.1, volatility: 0.5, replicas: 4), 1.0)

      expect(value).to be_within(1e-4).of(normal_tail((4.0 - 7.0) / (0.5 * Math.sqrt(30.0))))
    end

    it "puts the probability that others changed nothing on the assumed amount" do
      # over 30 s of a 60 s scale, quiet with probability 0.64^0.5 = 0.8; if
      # not, the mixture's mean 0.1 * 30 = 3 and variance 0.25 * 30 = 7.5
      # come from a normal of mean 3 / 0.2 = 15 and variance
      # (7.5 + 9) / 0.2 - 225 = -142.5, i.e. 0 (all the growth is the drift)
      quiet = 0.64
      age = 30.0
      p0 = Math.sqrt(quiet)
      shift = 0.1 * age / (1.0 - p0)
      spread = Math.sqrt([((0.25 * age) + ((0.1 * age)**2)) / (1.0 - p0) - (shift**2), 0.0].max)
      expected = ->(needed, free) do
        changed = if spread.positive?
          normal_tail((needed - (free - shift)) / spread)
        else
          (((free - shift) >= needed) ? 1.0 : 0.0)
        end
        (p0 * ((free >= needed) ? 1.0 : 0.0)) + ((1.0 - p0) * changed)
      end

      expect(cpu_fit(observation(free: 10.0, age: age, drift: 0.1, volatility: 0.5, replicas: 4, quiet: quiet), 1.0)).to be_within(1e-4).of(expected.call(4.0, 10.0))
      # 5 needed of 4.5: only if something changed (and it did not help)
      expect(cpu_fit(observation(free: 4.5, age: age, drift: -0.1, volatility: 2.0, replicas: 5, quiet: quiet), 1.0)).to be < 0.2
    end

    it "is certain for a fresh reading and for no replicas" do
      fresh = observation(free: 3.5, age: 0.0, drift: 0.1, volatility: 0.5, replicas: 4)

      expect(cpu_fit(fresh, 1.0)).to be == 0.0
      expect(cpu_fit(observation(free: 4.0, age: 0.0, drift: 0.0, volatility: 0.5, replicas: 4), 1.0)).to be_within(1e-9).of(1.0)
      expect(cpu_fit(fresh, 0.0)).to be_within(1e-9).of(1.0)
    end

    it "is differentiable in the shares" do
      shares = Torch.tensor([[0.5, 0.3, 0.2]]).requires_grad!(true)
      safety.score(observation(free: 3.0, age: 20.0, drift: 0.0, volatility: 0.3, replicas: 5), shares).sum.backward

      expect(shares.grad.abs.sum.item).to be > 0.0
    end

    it "is calibrated on the monitored testbed" do
      srand(2)
      environment = Reinforce::Environments::MultiCluster.monitored(clusters: 4, horizon: 600)
      safety = subject.new(environment)
      calibration = Reinforce::Shields::Calibration.new
      observation = environment.reset
      done = false
      until done
        replicas = environment.request[:replicas]
        before = Torch.tensor([observation], dtype: :float32)
        observation, _reward, done, info = environment.step(environment.action_space.sample)
        next if info[:deployments].empty?

        shares = Array.new(5, 0.0)
        info[:deployments].each { |deployment| shares[deployment[:cluster]] = deployment[:replicas].to_f / replicas }
        shares[4] = 1.0 - shares.sum
        probabilities = safety.fit_probabilities(before, Torch.tensor([shares], dtype: :float32))[0].to_a
        info[:deployments].each { |deployment| calibration.add(probabilities[deployment[:cluster]], deployment[:fitted]) }
      end

      expect(calibration.count).to be > 1000
      expect(calibration.expected_calibration_error).to be < 0.03
    end

    it "gives fit probabilities per cluster" do
      probabilities = safety.fit_probabilities(observation(free: 10.0, age: 30.0, drift: 0.1, volatility: 0.5, replicas: 4),
        Torch.tensor([[1.0, 0.0, 0.0]]))

      expect(probabilities.shape).to be == [1, 2]
      expect(probabilities[0, 1].item).to be_within(1e-9).of(1.0)
    end
  end
end
