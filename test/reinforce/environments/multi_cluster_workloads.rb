# frozen_string_literal: true

require "reinforce"

describe Reinforce::Environments::MultiCluster::Workloads::Diurnal do
  let(:environment_class) { Reinforce::Environments::MultiCluster }
  let(:config) { environment_class.diurnal(clusters: 4).config }

  it "sets the mean lifetime by Little's law for the most demanded resource" do
    # per request: 5.5 replicas x 0.2 GPU = 1.1 of the 4 x 8 GPUs, the
    # largest fraction of any resource
    expect(config.workload.mean_lifetime(config)).to be_within(1e-9).of(0.85 / (1.1 / 32))
  end

  it "occupies mean_load of the capacity on average when everything is deployed" do
    srand(1)
    workload = subject.new(amplitude: 0.4, period: 600.0)
    run = workload.start(config, Random.new(1))
    capacity = Array.new(4, 1e9)
    free = [capacity.dup]
    occupied_time = 0.0
    last_time = 0.0
    20_000.times do
      request = run.next_request
      amounts = environment_class::RESOURCES.map { |resource| request[:replicas] * request[:demand][resource] }
      run.deployed(0, amounts, request)
      amounts.each_with_index { |amount, resource| free[0][resource] -= amount }
      occupied = 1e9 - free[0][2]
      run.advance(free, capacity)
      occupied_time += occupied * (run.time - last_time)
      last_time = run.time
    end
    expected = 0.85 * 8 * 4

    expect(occupied_time / last_time).to be_within(0.05 * expected).of(expected)
  end

  it "makes arrivals more frequent while the rate is above its mean" do
    workload = subject.new(amplitude: 0.4, period: 100.0)
    run = workload.start(config, Random.new(2))
    free = [Array.new(4, 0.0)]
    counts = Hash.new(0)
    20_000.times do
      run.advance(free, Array.new(4, 0.0))
      counts[Math.sin(2.0 * Math::PI * run.time / 100.0).positive?] += 1
    end
    # rate 1 +- 0.4 * (mean of |sin|) over the two halves of the period
    expected = (1.0 + (0.8 / Math::PI)) / (1.0 - (0.8 / Math::PI))

    expect(counts[true].to_f / counts[false]).to be_within(0.1 * expected).of(expected)
    expect(20_000 / run.time).to be_within(0.03).of(1.0)
  end

  it "gives back exactly the resources of departed services" do
    srand(3)
    environment = environment_class.diurnal(clusters: 2, horizon: 10_000)
    environment.reset
    300.times { environment.step([1.0, 1.0, -1.0]) }
    peak = environment.free.map(&:dup)
    2_000.times { environment.step([-1.0, -1.0, 1.0]) }

    expect(peak).not.to be == environment.free
    expect(environment.free).to be == Array.new(2) { environment.capacities }
    expect(environment.workload.active).to be == 0
  end

  it "reports the offered load and the phase" do
    srand(4)
    environment = environment_class.diurnal(clusters: 2)
    observation = environment.reset
    _observation, _reward, _done, info = environment.step([0.0, 0.0, 0.0])
    phase = observation[environment.observation_layout.workload, 2]

    expect(environment.state_size).to be == environment.observation_layout.workload + 2
    expect(phase).to be == [0.5, 1.0]
    expect(info[:metrics][:offered_load]).to be_within(0.85 * 0.4).of(0.85)
    expect(info[:metrics][:utilization]).to be_within(0.5).of(0.5)
    expect(environment_class.diurnal(clusters: 2, workload: {observe_phase: false}).state_size).to be == environment.state_size - 2
  end

  it "samples requests from weighted classes" do
    classes = [
      Reinforce::Environments::MultiCluster::Workloads::RequestClass.new(weight: 3.0, replicas: 1..1, gpu_demand: {0 => 1.0}),
      Reinforce::Environments::MultiCluster::Workloads::RequestClass.new(weight: 1.0, replicas: 2..2, gpu_demand: {1 => 1.0}, lifetime_scale: 2.0)
    ]
    run = subject.new(classes: classes).start(config, Random.new(5))
    requests = Array.new(4_000) { run.next_request }
    gpu = requests.select { |request| request[:demand][:gpu] == 1 }

    expect(gpu.map { |request| request[:replicas] }.uniq).to be == [2]
    expect(gpu.size / 4_000.0).to be_within(0.03).of(0.25)
    expect(gpu.sum { |request| request[:lifetime] } / gpu.size).to be > (requests - gpu).sum { |request| request[:lifetime] } / (requests - gpu).size
  end

  it "keeps classes within the environment's limits" do
    oversized = Reinforce::Environments::MultiCluster::Workloads::RequestClass.new(replicas: 1..100)

    expect { subject.new(classes: [oversized]).mean_lifetime(config) }.to raise_exception(ArgumentError, message: be =~ /limits/)
    expect { subject.new(amplitude: 1.0) }.to raise_exception(ArgumentError)
  end
end

describe Reinforce::Environments::MultiCluster do
  it "keeps one request per step and random recovery by default" do
    srand(6)
    environment = subject.new(clusters: 2)
    environment.reset
    _observation, _reward, _done, info = environment.step([0.0, 0.0, 0.0])

    expect(environment.workload.time).to be == 1
    expect(info[:metrics].key?(:offered_load)).to be == false
    expect(environment.release_bound).to be == {cpu: 25.0, memory: 50.0, gpu: 1.0, bandwidth: 500.0}
    expect(environment.telemetry_delay_bound).to be == 1
  end

  it "shows each cluster's last scrape with telemetry_interval" do
    srand(7)
    environment = subject.diurnal(clusters: 2, telemetry_interval: 10.0)
    environment.reset
    differences = 0
    40.times do
      observation, = environment.step([1.0, -1.0, -1.0])
      measured = environment.decode(observation)[:free]
      phases = environment.instance_variable_get(:@scrape_phase)
      now = environment.workload.time
      history = environment.instance_variable_get(:@history)
      measured.each_with_index do |free, cluster|
        scrape = now - ((now - phases[cluster]) % 10.0)
        expected = (history.rfind { |time, _free| time <= scrape } || history.first)[1][cluster]
        expect(free.zip(expected).all? { |a, b| (a - b).abs < 1e-6 }).to be == true
      end
      differences += 1 if measured.flatten.zip(environment.free.flatten).any? { |a, b| (a - b).abs > 1e-6 }
    end

    expect(differences).to be > 0
    # the arrivals expected in one interval at the peak rate
    expect(environment.telemetry_delay_bound).to be == 14
  end
end
