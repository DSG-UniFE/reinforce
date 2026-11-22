# frozen_string_literal: true

require "tmpdir"
require "reinforce"

describe Reinforce::Environments::MultiCluster::Profile do
  def rates(profile, times, seed: 1)
    run = profile.start(Random.new(seed))
    times.map { |time| run.rate(time) }
  end

  it "follows its segments, then holds its final value" do
    profile = subject.build do
      constant 2.0, duration: 10
      ramp to: 4.0, duration: 10
      sine mean: 1.0, amplitude: 0.5, period: 40, duration: 40
    end

    expect(profile.duration).to be_within(1e-9).of(60.0)
    values = rates(profile, [0, 9.9, 15, 20, 30, 50, 100])
    expect(values[0..2]).to be == [2.0, 2.0, 3.0]
    expect(values[3]).to be_within(1e-9).of(1.0)
    expect(values[4]).to be_within(1e-9).of(1.5)
    expect(values[5]).to be_within(1e-9).of(0.5)
    expect(values[6]).to be_within(1e-9).of(1.0)
    expect(profile.maximum).to be_within(1e-9).of(4.0)
  end

  it "starts over when cyclic" do
    profile = subject.build(cyclic: true) do
      constant 1.0, duration: 10
      constant 3.0, duration: 10
    end

    expect(rates(profile, [5, 15, 25, 35])).to be == [1.0, 3.0, 1.0, 3.0]
  end

  it "interpolates a trace, inline or from a CSV file" do
    inline = subject.build { trace rows: [[100, 1.0], [110, 3.0], [130, 1.0]], scale: 2.0 }
    expect(inline.duration).to be_within(1e-9).of(30.0)
    expect(rates(inline, [0, 5, 10, 20, 40])).to be == [2.0, 4.0, 6.0, 4.0, 2.0]

    Dir.mktmpdir do |directory|
      path = File.join(directory, "trace.csv")
      File.write(path, "time,other,rate\n0,9,1.0\n10,9,3.0\n")
      from_file = subject.build { trace path }
      expect(rates(from_file, [5])).to be == [2.0]
    end
  end

  it "adds spikes on top of the segments" do
    profile = subject.build do
      constant 1.0, duration: 100
      spike peak: 3.0, at: 50, width: 5
    end

    values = rates(profile, [0, 50, 55])
    expect(values[0]).to be_within(1e-9).of(1.0)
    expect(values[1]).to be_within(1e-9).of(4.0)
    expect(values[2]).to be_within(1e-9).of(1.0 + (3.0 * Math.exp(-0.5)))
    expect(profile.maximum).to be_within(1e-9).of(4.0)
  end

  it "multiplies by a correlated log-normal noise of mean 1, within its maximum" do
    profile = subject.build do
      constant 2.0, duration: 1
      noise sigma: 0.3, correlation: 10
    end
    values = rates(profile, (0...20_000).map { |i| i * 2.5 })

    expect(values.sum / values.size).to be_within(0.05).of(2.0)
    expect(values.max).to be <= profile.maximum
    # consecutive values, a quarter of the correlation time apart, move by
    # 0.3 * E|x' - x| = 0.16 in log on average, against 0.34 if independent
    jumps = values.each_cons(2).sum { |a, b| (Math.log(b) - Math.log(a)).abs } / (values.size - 1)
    expect(jumps).to be_within(0.02).of(0.3 * Math.sqrt(2.0 * (1.0 - Math.exp(-0.25))) * Math.sqrt(2.0 / Math::PI))
    expect(rates(profile, [1000.0], seed: 1)).to be == rates(profile, [1000.0], seed: 1)
  end

  it "reads a specification, e.g. from YAML" do
    spec = [{"constant" => 0.5, "duration" => 10}, {"ramp" => {"to" => 1.5, "duration" => 10}}, {"spike" => {"peak" => 1, "at" => 5, "width" => 1}}]
    profile = subject.from_spec(spec)

    expect(rates(profile, [15])).to be == [1.0]
    expect(profile.spikes.size).to be == 1
    expect { subject.from_spec([{"wobble" => 1}]) }.to raise_exception(ArgumentError)
  end
end

describe Reinforce::Environments::MultiCluster::Workloads::Scripted do
  let(:environment_class) { Reinforce::Environments::MultiCluster }
  let(:profile_class) { Reinforce::Environments::MultiCluster::Profile }
  let(:config) { environment_class.new(clusters: 2).config }

  def capacity = config.capacity.values_at(:cpu, :memory, :gpu, :bandwidth).map(&:to_f)

  it "makes arrivals follow the profile" do
    profile = profile_class.build do
      constant 0.5, duration: 1000
      constant 2.0, duration: 1000
    end
    run = subject.new(profile: profile).start(config, Random.new(1))
    free = Array.new(2) { capacity }
    times = []
    times << run.time while run.advance(free, capacity) && run.time < 2000

    first = times.count { |time| time < 1000 }
    expect(first).to be_within(60).of(500)
    expect(times.size - first).to be_within(150).of(2000)
  end

  it "gives back its services' resources, reporting each change before applying it" do
    run = subject.new(mean_lifetime: 10.0).start(config, Random.new(2))
    free = Array.new(2) { capacity }
    request = run.next_request
    amounts = [1.0, 2.0, 0.0, 3.0]
    free[1] = free[1].each_with_index.map { |value, resource| value - amounts[resource] }
    run.deployed(1, amounts, request)
    events = []
    run.advance(free, capacity) { |event| events << [event, free[1].dup] } until events.any?

    event, before = events.first
    expect(event.kind).to be == :departure
    expect(event.own).to be == true
    expect(event.cluster).to be == 1
    expect(event.amounts).to be == amounts
    expect(event.time).to be_within(1e-9).of(request[:lifetime])
    # reported before being applied
    expect(before).to be == capacity.each_with_index.map { |value, resource| value - amounts[resource] }
    expect(free[1]).to be == capacity
  end

  it "runs background tenants that occupy their mean load" do
    # one-replica services: GPU is the most demanded resource (0.2 on
    # average of a cluster's 2, against 3 of 50 CPUs, 5.5 of 100 GB and 50.5
    # of 1000 bandwidth units)
    background = Reinforce::Environments::MultiCluster::Workloads::Background.new(mean_load: 0.3, mean_lifetime: 50.0, replicas: 1..1)
    run = subject.new(profile: profile_class.constant(1.0), background: background).start(config, Random.new(3))
    free = Array.new(2) { capacity }
    occupied = 0.0
    last = 0.0
    kinds = Hash.new(0)
    # the time average of the GPUs in use on cluster 0, integrated exactly
    # between changes (reported before they are applied)
    60_000.times do
      run.advance(free, capacity) do |event|
        kinds[[event.kind, event.own]] += 1
        next unless event.cluster == 0

        occupied += (capacity[2] - free[0][2]) * (event.time - last)
        last = event.time
      end
      occupied += (capacity[2] - free[0][2]) * (run.time - last)
      last = run.time
    end

    expect(kinds.keys.sort).to be == [[:arrival, false], [:departure, false]]
    # 0.3 of 2 GPUs is 0.6 GPUs offered to 2 "servers": an Erlang loss
    # system, which drops (A^2/2) / (1 + A + A^2/2) = 10% of them, so 0.27 is
    # carried
    expect(occupied / last / capacity[2]).to be_within(0.03).of(0.27)
    expect(run.background_dropped).to be > 0
  end

  it "drives the environment, one decision per arrival" do
    srand(4)
    workload = subject.new(profile: profile_class.constant(2.0), mean_lifetime: 30.0,
      background: Reinforce::Environments::MultiCluster::Workloads::Background.new(mean_load: 0.2))
    environment = environment_class.new(clusters: 2, horizon: 400, workload: workload)
    environment.reset
    done = false
    steps = 0
    until done
      _observation, _reward, done, info = environment.step([1.0, 1.0, -1.0])
      steps += 1
    end

    expect(steps).to be == 400
    expect(info[:metrics][:served]).to be >= 0
    expect(environment.workload.time).to be_within(40).of(200)
  end
end

describe Reinforce::Environments::MultiCluster::Workloads::Turnover do
  it "reports each cluster's recovery before applying it" do
    config = Reinforce::Environments::MultiCluster.new(clusters: 2).config
    run = subject.new.start(config, Random.new(5))
    capacity = [50.0, 100.0, 2.0, 1000.0]
    free = [[0.0] * 4, [0.0] * 4]
    events = []
    run.advance(free, capacity) { |event| events << event }

    expect(events.map { |event| [event.cluster, event.kind, event.own, event.time] }).to be == [[0, :release, false, 1], [1, :release, false, 1]]
    expect(free).to be == events.map(&:amounts)
  end
end
