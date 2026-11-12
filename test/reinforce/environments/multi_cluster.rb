# frozen_string_literal: true

require "reinforce"

describe Reinforce::Environments::MultiCluster do
  # One cluster's free resources, as [cpu, memory, gpu, bandwidth].
  def free_of(environment, cluster)
    environment.free[cluster]
  end

  # Replaces the current request (and the measurements the agent sees).
  def set_request(environment, replicas:, demand:, dependency: nil)
    environment.instance_variable_set(:@request, {replicas: replicas, demand: demand, dependency: dependency})
  end

  # An action giving each cluster (and the reject option) the given share.
  def action_for(shares)
    shares.map { |share| (2.0 * share) - 1.0 }
  end

  let(:environment) do
    srand(7)
    subject.new(clusters: 3, max_delay: 0, horizon: 5)
  end

  it "has an observation and action space sized by the number of clusters" do
    observation = environment.reset

    expect(environment.state_size).to be == (8 * 3) + 5
    expect(observation.size).to be == environment.state_size
    expect(observation.all? { |value| value.between?(0.0, 1.0) }).to be == true
    expect(environment.action_space).to be == Reinforce::Spaces::Box.new(low: -1.0, high: 1.0, size: 4)
    expect(environment.class.ancestors.include?(Reinforce::Environment)).to be == true
  end

  it "scales the default replica limit with the clusters" do
    expect(subject.new(clusters: 8).config.replica_limit).to be == 20
    expect(subject.new(clusters: 8, max_replicas: 6).config.replica_limit).to be == 6
  end

  it "turns actions into shares and integer counts" do
    expect(subject.shares([1.0, -1.0, 0.0])).to be == [2.0 / 3, 0.0, 1.0 / 3]
    expect(subject.shares([-1.0, -2.0])).to be == [0.5, 0.5]
    expect(subject.allocate(10, [0.25, 0.25, 0.5])).to be == [3, 2, 5]
    expect(subject.allocate(7, [1.0 / 3, 1.0 / 3, 1.0 / 3])).to be == [3, 2, 2]
    expect(subject.allocate(5, [0.0, 0.0, 1.0])).to be == [0, 0, 5]
  end

  it "deploys replicas that fit and consumes their resources" do
    environment.reset
    set_request(environment, replicas: 4, demand: {cpu: 5, memory: 10, gpu: 0, bandwidth: 100})
    _observation, reward, done, info = environment.step(action_for([1.0, 0.0, 0.0, 0.0]))

    expect(info[:metrics]).to have_keys(requested: be == 4, served: be == 4, rejected: be == 0, overflow: be == 0)
    expect(done).to be == false
    expect(reward).to be < 1.0
    # 50 - 20 CPU, then at most 25 recovered (capped at the capacity)
    expect(free_of(environment, 0)[0]).to be_within(25.0).of(42.5)
    expect(free_of(environment, 0)[0]).to be <= 50.0
  end

  it "deploys nothing on a cluster that overflows, counting the excess replicas" do
    environment.reset
    environment.free[1] = [12.0, 100.0, 2.0, 1000.0]
    set_request(environment, replicas: 4, demand: {cpu: 5, memory: 10, gpu: 0, bandwidth: 100})
    _observation, reward, _done, info = environment.step(action_for([0.0, 1.0, 0.0, 0.0]))

    # 2 replicas fit in 12 CPU: the other 2 overflow, and none is deployed
    expect(info[:metrics]).to have_keys(served: be == 0, overflow: be == 2, cost: be == 0.0)
    expect(reward).to be_within(1e-9).of(1.0 - (2.0 * 2 / 4))
  end

  it "rewards rejection, overflow, cost and latency with their weights" do
    environment.reset
    set_request(environment, replicas: 4, demand: {cpu: 1, memory: 1, gpu: 0, bandwidth: 1}, dependency: 2)
    _observation, reward, _done, info = environment.step(action_for([0.5, 0.0, 0.0, 0.5]))
    prices = environment.instance_variable_get(:@economic_price)[0] + environment.instance_variable_get(:@energy_price)[0]
    latency = environment.latency_matrix[0][2]

    expect(info[:metrics]).to have_keys(served: be == 2, rejected: be == 2)
    expect(reward).to be_within(1e-9).of(1.0 - (2.0 * 0.5) - (0.5 * prices / 2.0) - (0.5 * latency))
  end

  it "truncates episodes at the horizon" do
    environment.reset
    results = Array.new(5) { environment.step(action_for([0.0, 0.0, 0.0, 1.0])) }

    expect(results.map { |result| result[2] }).to be == [false, false, false, false, true]
    expect(results.last[3][:truncated]).to be == true
  end

  it "samples a symmetric, normalized latency matrix once" do
    matrix = environment.latency_matrix
    environment.reset

    expect(environment.latency_matrix).to be_equal(matrix)
    expect(matrix.each_index.all? { |i| matrix[i][i].zero? && matrix[i] == matrix.map { |row| row[i] } }).to be == true
    expect(matrix.flatten.max).to be_within(1e-12).of(1.0)
  end

  it "decodes what an observation says" do
    observation = environment.reset
    request = environment.request
    decoded = environment.decode(observation)

    expect(decoded[:replicas]).to be == request[:replicas]
    expect(decoded[:demand]).to be == subject::RESOURCES.map { |resource| request[:demand][resource] }
    expect(decoded[:dependency]).to be == request[:dependency]
    expect(decoded[:free].flatten.zip(environment.free.flatten).all? { |a, b| (a - b).abs < 1e-9 }).to be == true
  end

  it "delays measurements by up to max_delay steps" do
    srand(3)
    delayed = subject.new(clusters: 2, max_delay: 3, horizon: 50)
    delayed.reset
    stale = 0
    20.times do
      observation, = delayed.step([1.0, 1.0, -1.0])
      decoded = delayed.decode(observation)
      stale += 1 if decoded[:free].flatten.zip(delayed.free.flatten).any? { |a, b| (a - b).abs > 1e-6 }
    end

    expect(stale).to be > 0
  end

  it "comes in an easy and a hard configuration" do
    expect(subject.hard.config.max_demand).to be == {cpu: 10, memory: 20, bandwidth: 200}
    expect(subject.easy(clusters: 8).config.clusters).to be == 8
    expect { environment.step([0.0]) }.to raise_exception(ArgumentError)
  end

  with "the greedy policy" do
    let(:policy) { Reinforce::Environments::MultiCluster::GreedyPolicy.new(environment) }

    it "puts every replica on the tightest cluster that hosts them all" do
      environment.reset
      environment.free[0] = [50.0, 100.0, 2.0, 1000.0]
      environment.free[1] = [30.0, 100.0, 2.0, 1000.0]
      environment.free[2] = [10.0, 100.0, 2.0, 1000.0]
      set_request(environment, replicas: 4, demand: {cpu: 5, memory: 10, gpu: 0, bandwidth: 100})
      environment.instance_variable_set(:@measured, environment.free.map(&:dup))
      observation = environment.send(:observe)

      expect(policy.predict(observation)).to be == [-1.0, 1.0, -1.0, -1.0]
    end

    it "spreads the replicas over the freest clusters and rejects the rest" do
      environment.reset
      environment.free[0] = [10.0, 100.0, 2.0, 1000.0]
      environment.free[1] = [5.0, 100.0, 2.0, 1000.0]
      environment.free[2] = [0.0, 100.0, 2.0, 1000.0]
      set_request(environment, replicas: 5, demand: {cpu: 5, memory: 10, gpu: 0, bandwidth: 100})
      environment.instance_variable_set(:@measured, environment.free.map(&:dup))
      observation = environment.send(:observe)
      counts = policy.predict(observation).map { |value| ((value + 1.0) * 5 / 2.0).round }

      expect(counts).to be == [2, 1, 0, 2]
    end
  end
end
