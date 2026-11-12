# frozen_string_literal: true

require "reinforce/shields"
require_relative "../../support/torch_test_helpers"

# The probability that an allocation is safe under the probabilistic logic
# program the safety formula stands for, by enumerating its possible worlds:
#
#   p_cpu(C)::cpu_fit(C).  p_mem(C)::mem_fit(C).  p_gpu(C)::gpu_fit(C).  p_bw(C)::bw_fit(C).
#   h(C)::comfortable_if_fit(C).  j::reject_safe.
#   pi(C)::pick(C).        % pi(c_i) = q_i / (q_i + ... + q_n + q_reject)
#   chosen(c_i) :- \+pick(c_1), ..., \+pick(c_i-1), pick(c_i).
#   chosen(reject) :- \+pick(c_1), ..., \+pick(c_n).
#   full_fit(C) :- cpu_fit(C), mem_fit(C), gpu_fit(C), bw_fit(C).
#   tight(C) :- full_fit(C), \+comfortable_if_fit(C).
#   unsafe :- chosen(C), \+full_fit(C).
#   rho::unsafe :- chosen(C), tight(C).
#   unsafe :- chosen(reject), \+reject_safe.
#   safe :- \+unsafe.
def program_probability(shares, fit, comfortable, reject_safe, tight_penalty)
  clusters = fit.size
  remaining = shares.sum
  pick = shares.first(clusters).map do |share|
    probability = remaining.positive? ? share / remaining : 0.0
    remaining -= share
    probability
  end
  facts = fit.flatten + comfortable + [reject_safe] + pick + Array.new(clusters, tight_penalty)

  (0...(1 << facts.size)).sum do |world|
    holds = facts.each_index.map { |index| world[index] == 1 }
    weight = facts.each_with_index.reduce(1.0) { |product, (probability, index)| product * (holds[index] ? probability : 1.0 - probability) }
    fits = Array.new(clusters) { |cluster| holds[cluster * 4, 4].all? }
    comfortable_holds = holds[clusters * 4, clusters]
    picks = holds[(clusters * 5) + 1, clusters]
    rho = holds[(clusters * 6) + 1, clusters]
    chosen = picks.index(true)
    safe = if chosen
      fits[chosen] && !(rho[chosen] && !comfortable_holds[chosen])
    else
      holds[clusters * 5]
    end
    safe ? weight : 0.0
  end
end

describe Reinforce::Shields::MultiClusterSafety do
  def build_environment(**options)
    srand(5)
    Reinforce::Environments::MultiCluster.new(clusters: 2, **options)
  end

  let(:environment) { build_environment }
  let(:safety) { subject.new(environment) }

  # An observation with the given measured free resources (clusters x
  # [cpu, memory, gpu, bandwidth]), replica count and per-replica demand.
  def observation(environment, free:, replicas:, demand:, dependency: nil, latency: [0.0, 0.0])
    layout = environment.observation_layout
    capacity = environment.capacities
    max_demand = environment.config.max_demand_per_resource
    values = Array.new(layout.size, 0.0)
    free.each_with_index do |cluster, index|
      cluster.each_with_index { |amount, resource| values[layout.free + (index * 4) + resource] = amount / capacity[resource] }
    end
    latency.each_with_index { |value, index| values[layout.latency + index] = value }
    values[layout.dependency + dependency] = 1.0 if dependency
    values[layout.replicas] = replicas.to_f / environment.config.replica_limit
    Reinforce::Environments::MultiCluster::RESOURCES.each_with_index do |resource, index|
      values[layout.demand + index] = demand[index].to_f / max_demand[resource]
    end
    Torch.tensor([values], dtype: :float32)
  end

  it "is the probability of safety under the probabilistic logic program" do
    TorchTestHelpers.with_torch_seed do
      cases = Array.new(5) do
        shares = Array.new(3) { rand }
        total = shares.sum
        {shares: shares.map { |share| share / total }, fit: Array.new(2) { Array.new(4) { rand } },
         comfortable: Array.new(2) { rand }, reject_safe: rand}
      end
      # zero shares, and a certain fit
      cases << {shares: [0.0, 0.7, 0.3], fit: [[1.0, 1.0, 1.0, 1.0], [0.9, 1.0, 0.5, 1.0]], comfortable: [0.2, 0.0], reject_safe: 0.4}

      cases.each do |example|
        score = safety.evaluate(
          chosen: Torch.tensor([example[:shares].first(2)]), fit: Torch.tensor([example[:fit]]),
          comfortable: Torch.tensor([example[:comfortable]]), reject_chosen: Torch.tensor([example[:shares].last]),
          reject_safe: Torch.tensor([example[:reject_safe]])
        )
        expected = program_probability(example[:shares], example[:fit], example[:comfortable], example[:reject_safe], 0.3)

        expect(score.to_a.first).to be_within(1e-5).of(expected)
      end
    end
  end

  it "scores a batch of allocations at once" do
    chosen = Torch.tensor([[0.5, 0.5], [1.0, 0.0], [0.0, 0.0]])
    fit = Torch.ones(3, 2, 4)
    comfortable = Torch.tensor([[1.0, 0.0], [0.5, 1.0], [1.0, 1.0]])
    scores = safety.evaluate(chosen: chosen, fit: fit, comfortable: comfortable,
      reject_chosen: Torch.tensor([0.0, 0.0, 1.0]), reject_safe: Torch.tensor([1.0, 1.0, 0.25]))

    # 0.5 * 1 + 0.5 * (1 - 0.3); 1 - 0.3 * 0.5; the reject share times reject_safe
    expect(scores.to_a.zip([0.85, 0.85, 0.25]).all? { |score, expected| (score - expected).abs < 1e-6 }).to be == true
  end

  it "reads fits as certain without delays" do
    environment = build_environment(max_delay: 0)
    safety = subject.new(environment)
    observations = observation(environment, free: [[20, 100, 2, 1000], [9, 100, 2, 1000]], replicas: 4, demand: [5, 10, 0, 100])
    sensors = safety.sensors(observations, Torch.tensor([[0.5, 0.5, 0.0]]))

    # 2 replicas need 10 CPU: cluster 0 has 20, cluster 1 only 9
    expect(sensors[:fit][0].to_a).to be == [[1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0]]
    # 10 / 0.8 = 12.5 CPU still fit in cluster 0
    expect(sensors[:comfortable][0].to_a).to be == [1.0, 0.0]
    expect(safety.score(observations, Torch.tensor([[0.5, 0.5, 0.0]])).to_a.first).to be_within(1e-6).of(0.5)
  end

  it "spreads the measurement over the availability a delay allows" do
    # one step of delay: CPU between 30 - 5 * 5 = 5 and min(50, 30 + 25) = 50
    observations = observation(environment, free: [[30, 100, 2, 1000], [30, 100, 2, 1000]], replicas: 4, demand: [5, 1, 0, 1])
    fit = safety.sensors(observations, Torch.tensor([[1.0, 0.0, 0.0]]))[:fit]

    # 20 CPU: certain now; after a step, (50 - 20) / (50 - 5) of the time
    expect(fit[0, 0, 0].item).to be_within(1e-6).of((1.0 + (30.0 / 45.0)) / 2.0)
  end

  it "justifies only the rejections the clusters' capacity could not host" do
    environment = build_environment(max_delay: 0)
    safety = subject.new(environment)
    # each cluster hosts 2 replicas (10 CPU each), so 1 of 5 must be rejected
    observations = observation(environment, free: [[10, 100, 2, 1000], [10, 100, 2, 1000]], replicas: 5, demand: [5, 1, 0, 1])
    rejecting_two = safety.sensors(observations, Torch.tensor([[0.3, 0.3, 0.4]]))[:reject_safe].item
    rejecting_none = safety.sensors(observations, Torch.tensor([[0.5, 0.5, 0.0]]))[:reject_safe].item

    # the bisection estimates 1.97 replicas per cluster (6 steps over [0, 5])
    expect(rejecting_two).to be_within(1e-3).of((5.0 - (2 * 1.953125)) / 2.0)
    expect(rejecting_none).to be_within(1e-6).of(1.0)
  end

  it "prefers allocations close to the dependency" do
    observations = observation(environment, free: [[50, 100, 2, 1000]] * 2, replicas: 2, demand: [1, 1, 0, 1], dependency: 0, latency: [0.0, 0.8])
    latencies = safety.latency(Torch.cat([observations] * 3), Torch.tensor([[1.0, 0.0, 0.0], [0.25, 0.25, 0.5], [0.0, 0.0, 1.0]]))
    independent = observation(environment, free: [[50, 100, 2, 1000]] * 2, replicas: 2, demand: [1, 1, 0, 1])

    expect(latencies.to_a.map { |value| value.round(6) }).to be == [0.0, 0.4, 1.0]
    expect(safety.latency(independent, Torch.tensor([[0.0, 0.0, 1.0]])).to_a).to be == [0.0]
  end

  it "is differentiable in the shares" do
    observations = observation(environment, free: [[12, 100, 2, 1000], [40, 100, 2, 1000]], replicas: 4, demand: [5, 1, 0, 1])
    shares = Torch.tensor([[0.6, 0.3, 0.1]]).requires_grad!(true)
    safety.score(observations, shares).sum.backward

    # moving replicas from the full cluster to the free one is safer
    expect(shares.grad[0, 1].item).to be > shares.grad[0, 0].item
  end
end
