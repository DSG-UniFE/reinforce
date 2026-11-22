# frozen_string_literal: true

require "reinforce"

describe Reinforce::Environments::MultiCluster::Telemetry do
  let(:capacity) { [50.0, 100.0, 2.0, 1000.0] }

  def monitor(telemetry, free: [capacity.dup], seed: 1)
    telemetry.start(free, capacity, Random.new(seed))
  end

  it "scrapes each cluster periodically and uses the latest reading available" do
    telemetry = subject.new(interval: 10.0, jitter: 0.0, latency_median: 2.0, latency_sigma: 0.0)
    free = [capacity.dup]
    pipeline = monitor(telemetry, free: free)
    known = (1..100).map do |second|
      # the scrapes up to `second` see the state set at the previous second
      pipeline.advance(second.to_f, free)
      free[0][0] = 50.0 - second
      [second, pipeline.knowledge(second.to_f).first]
    end
    readings = known.map(&:last).uniq(&:scraped_at).drop(1) # after the initial one

    expect(readings.each_cons(2).map { |a, b| (b.scraped_at - a.scraped_at).round(6) }.uniq).to be == [10.0]
    readings.each { |reading| expect(reading.reading[0]).to be_within(1e-9).of(50.0 - reading.scraped_at.floor) }
    known.each do |second, reading|
      expect(reading.age).to be_within(1e-9).of(second - reading.scraped_at)
      # available 2 s after the scrape
      expect(reading.age).to be >= 2.0 if reading.scraped_at.positive?
    end
  end

  it "corrects readings with the agent's own changes since them" do
    telemetry = subject.new(interval: 10.0, jitter: 0.0, latency_median: 0.0, latency_sigma: 0.0)
    pipeline = monitor(telemetry)
    phase = pipeline.knowledge(0.0).first.scraped_at
    free = [capacity.dup]
    pipeline.advance(5.0, free)
    scraped = pipeline.knowledge(5.0).first.scraped_at
    pipeline.record_own(scraped - 0.5, 0, [7.0, 0.0, 0.0, 0.0]) # before the scrape: in the reading
    pipeline.record_own(scraped, 0, [1.0, 0.0, 0.0, 0.0]) # at the scrape's time: not in it
    pipeline.record_own(scraped + 1.0, 0, [2.0, 3.0, 0.0, 0.0])
    pipeline.record_own(scraped + 2.0, 0, [-0.5, 0.0, 0.0, 0.0]) # a departure

    expect(phase).to be == 0.0
    expect(pipeline.knowledge(scraped + 3.0).first.ledger).to be == [2.5, 3.0, 0.0, 0.0]
  end

  it "keeps the last reading when scrapes fail, which then gets older" do
    pipeline = monitor(subject.new(interval: 5.0, failure: 1.0))
    free = [capacity.map { |amount| amount / 2 }]
    pipeline.advance(100.0, free)
    known = pipeline.knowledge(100.0).first

    expect(known.scraped_at).to be == 0.0
    expect(known.age).to be_within(1e-9).of(100.0)
    expect(known.reading).to be == capacity
  end

  it "estimates the drift and volatility of other tenants' consumption" do
    telemetry = subject.new(interval: 10.0, jitter: 0.0, latency_median: 0.0, latency_sigma: 0.0, smoothing: 0.05)
    random = Random.new(7)
    # far from 0 and the capacity, so that nothing is censored
    free = [[1e5, 50.0, 1.0, 500.0]]
    pipeline = monitor(telemetry, free: free)
    # CPU: others take 0.05 per second on average, with a volatility of 0.2
    # per square root of a second (a random walk, sampled every second)
    (1..20_000).each do |second|
      pipeline.advance(second.to_f, free)
      normal = Math.sqrt(-2.0 * Math.log(1.0 - random.rand)) * Math.cos(2.0 * Math::PI * random.rand)
      free[0][0] -= 0.05 - (0.2 * normal)
    end
    known = pipeline.knowledge(20_000.0).first

    expect(known.drift[0]).to be_within(0.03).of(0.05)
    expect(known.volatility[0]).to be_within(0.05).of(0.2)
    expect(known.drift[1]).to be == 0.0
  end
  it "estimates how often other tenants change each resource" do
    telemetry = subject.new(interval: 10.0, jitter: 0.0, latency_median: 0.0, latency_sigma: 0.0, smoothing: 0.02)
    random = Random.new(8)
    free = [[25.0, 50.0, 1.0, 500.0]]
    pipeline = monitor(telemetry, free: free)
    # CPU changes 0.02 times per second (a Poisson process), up or down; nothing else
    (1..40_000).each do |second|
      pipeline.advance(second.to_f, free)
      free[0][0] += (random.rand < 0.5) ? 1.0 : -1.0 if random.rand < 0.02
    end
    rates = pipeline.knowledge(40_000.0).first.event_rate

    expect(rates[0]).to be_within(0.005).of(0.02)
    expect(rates[1]).to be_within(1e-9).of(0.0)
  end
end

describe Reinforce::Environments::MultiCluster do
  let(:workloads) { Reinforce::Environments::MultiCluster::Workloads }
  let(:telemetry_class) { Reinforce::Environments::MultiCluster::Telemetry }

  def environment(telemetry, background: nil, seed: 1)
    srand(seed)
    workload = workloads::Scripted.new(profile: Reinforce::Environments::MultiCluster::Profile.constant(1.0), mean_lifetime: 20.0,
      background: background)
    subject.new(clusters: 2, horizon: 300, workload: workload, telemetry: telemetry)
  end

  it "shows the true free resources when only the agent changes them and it keeps a ledger" do
    env = environment(telemetry_class.new(interval: 15.0))
    greedy = Reinforce::Environments::MultiCluster::GreedyPolicy.new(env)
    observation = env.reset
    overflowing = 0
    done = false
    until done
      decoded = env.decode(observation)
      decoded[:free].each_with_index do |cluster, index|
        cluster.each_with_index { |amount, resource| expect(amount).to be_within(1e-6).of(env.free[index][resource]) }
      end
      observation, _reward, done, info = env.step(greedy.predict(observation))
      overflowing += 1 if info[:deployments].any? { |deployment| !deployment[:fitted] }
    end

    # best fit on what the agent assumes never overflows
    expect(overflowing).to be == 0
  end

  it "shows raw readings without the ledger, which then go stale" do
    env = environment(telemetry_class.new(interval: 15.0, ledger: false))
    observation = env.reset
    differences = 0
    50.times do
      differences += 1 if env.decode(observation)[:free] != env.free
      observation, = env.step([1.0, 0.0, -1.0])
    end

    expect(differences).to be > 25
  end

  it "adds each cluster's reading age, and the others' drift and volatility, in [0, 1]" do
    background = workloads::Background.new(mean_load: 0.3, mean_lifetime: 60.0, replicas: 1..1)
    env = environment(telemetry_class.new(interval: 15.0), background: background)
    layout = env.observation_layout
    observation = env.reset
    ages = []
    metrics = Hash.new(0.0)
    100.times do
      observation, _reward, _done, info = env.step([1.0, 1.0, -1.0])
      ages.concat(observation[layout.age, 2])
      info[:metrics].each { |key, value| metrics[key] += value }
    end

    expect(observation.size).to be == layout.size
    expect(layout.size).to be == env.class.new(clusters: 2).observation_layout.size + 2 + 8 + 8 + 8
    expect(observation.all? { |value| value.between?(0.0, 1.0) }).to be == true
    # readings every 15 s, one decision a second, ages normalized by 60 s
    expect(ages.sum / ages.size).to be_within(0.06).of((7.5 + 2.0) / 60.0)
    expect(metrics.key?(:background_dropped)).to be == true
  end

  it "keeps the legacy telemetry when not configured, and refuses both" do
    legacy = subject.new(clusters: 2)

    expect(legacy.observation_layout.age).to be_nil
    expect { subject.new(telemetry: telemetry_class.new, telemetry_interval: 5.0) }.to raise_exception(ArgumentError)
  end
end
