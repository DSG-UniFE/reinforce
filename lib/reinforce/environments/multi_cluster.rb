# frozen_string_literal: true

require_relative "../environment"
require_relative "../spaces"

module Reinforce
  module Environments
    # Allocation of service replicas across the clusters of a Computing
    # Continuum infrastructure, with delayed resource telemetry.
    #
    # At each step a request asks for R identical replicas, each needing an
    # amount of CPU, memory, GPU and bandwidth, and possibly depending on a
    # service hosted by one of the clusters. The agent splits the replicas
    # among the clusters and a reject option:
    #
    # - **Actions** are vectors in [-1, 1]^(clusters + 1), the last
    #   component being rejection. Each component a becomes a weight
    #   clip((a + 1) / 2, 0, 1); normalized, the weights are the shares of
    #   the replicas (uniform if they are all 0), which are rounded to
    #   counts by the largest remainder method, so every replica goes to a
    #   cluster or is rejected.
    # - **Deployment**: a cluster's replicas are deployed only if all of
    #   them fit in every resource; otherwise none is, and those in excess
    #   of the most constrained resource count as overflow.
    # - **Dynamics**: after each step, every cluster recovers a random
    #   amount of each resource (running jobs completing), up to its
    #   capacity, and a new request arrives.
    # - **Observations** (#observation_layout) are in [0, 1]: the measured
    #   free resources of each cluster (normalized by capacity), economic
    #   and energy prices, the normalized latency from each cluster to the
    #   request's dependency and its one-hot location, the normalized
    #   replica count and per-replica demands. Each measurement is delayed
    #   by a random number of steps in 0..max_delay, drawn independently
    #   per cluster and resource.
    # - **Reward**: 1 - w1 * rejected/R - w2 * overflow/R - w3 * cost -
    #   w4 * latency, where cost and latency are averaged over the deployed
    #   replicas (prices and normalized latencies are in [0, 1]).
    #
    # Episodes last `horizon` steps and end truncated (`info[:truncated]`),
    # and `info[:metrics]` reports the step's requested, served, rejected
    # and overflowing replicas and the deployed replicas' total cost and
    # latency.
    class MultiCluster
      include Reinforce::Environment

      RESOURCES = %i[cpu memory gpu bandwidth].freeze

      # The environment's parameters. The defaults are the Easy
      # configuration (see .easy); .hard gives the Hard one.
      #
      # clusters: the number of clusters.
      # capacity: each cluster's capacity per resource.
      # max_demand: the maximum per-replica demand of CPU, memory and
      #   bandwidth, drawn uniformly in 1..max_demand.
      # gpu_demand: the distribution of the per-replica GPU demand, as
      #   {demand => probability}.
      # max_replicas: replica counts are drawn uniformly in 1..max_replicas;
      #   nil means 5/2 of the clusters (see #replica_limit), so the load per
      #   cluster does not depend on their number.
      # recovery: the maximum amount of CPU, memory and bandwidth each
      #   cluster recovers per step (uniformly from 0), and of GPU (an
      #   integer, uniformly from 0).
      # max_delay: measurement delays are drawn uniformly in 0..max_delay.
      # dependency_probability: the probability that a request depends on a
      #   service hosted by a (uniformly chosen) cluster.
      # horizon: the length of an episode.
      # reward_weights: the weights of rejection, overflow, cost and latency.
      Config = Data.define(:clusters, :capacity, :max_demand, :gpu_demand, :max_replicas, :recovery, :max_delay,
        :dependency_probability, :horizon, :reward_weights) do
        def initialize(clusters: 4, capacity: {cpu: 50, memory: 100, gpu: 2, bandwidth: 1000},
          max_demand: {cpu: 5, memory: 10, bandwidth: 100}, gpu_demand: {0 => 0.8, 1 => 0.2}, max_replicas: nil,
          recovery: {cpu: 25, memory: 50, gpu: 1, bandwidth: 500}, max_delay: 1, dependency_probability: 0.5,
          horizon: 1_000, reward_weights: [2.0, 2.0, 0.5, 0.5])
          super
        end

        # @returns [Integer] the maximum replica count of a request.
        def replica_limit
          max_replicas || (5 * clusters / 2)
        end

        # @returns [Hash{Symbol => Numeric}] the maximum per-replica demand of every resource.
        def max_demand_per_resource
          max_demand.merge(gpu: gpu_demand.keys.max)
        end
      end

      # Offsets of the observation's components (see #observation_layout).
      Layout = Data.define(:free, :economic_price, :energy_price, :latency, :dependency, :replicas, :demand, :size)

      # Small requests and fast recovery.
      def self.easy(**options)
        new(**options)
      end

      # Larger requests and slower recovery of CPU and bandwidth, so the
      # infrastructure cannot serve the whole workload at steady state.
      def self.hard(**options)
        new(max_demand: {cpu: 10, memory: 20, bandwidth: 200}, recovery: {cpu: 10, memory: 100, gpu: 1, bandwidth: 200}, **options)
      end

      attr_reader :config, :latency_matrix, :free, :request

      # @parameter config [Config, nil] the parameters, or nil for the defaults.
      # @parameter options [Hash] individual parameters, overriding `config`'s.
      def initialize(config = nil, **options)
        @config = config ? config.with(**options) : Config.new(**options)
        # Workload and delays have generators of their own, seeded from
        # Kernel.rand, so a seeded run is reproducible and the delays do not
        # perturb the workload.
        @random = Random.new(Kernel.rand(2**62))
        @delay_random = Random.new(Kernel.rand(2**62))
        @latency_matrix = sample_latency_matrix
      end

      def state_size
        observation_layout.size
      end

      def action_space
        Reinforce::Spaces::Box.new(low: -1.0, high: 1.0, size: clusters + 1)
      end

      # @returns [Integer] the number of clusters.
      def clusters
        @config.clusters
      end

      # @returns [Layout] the start of each component of the observation:
      #   `free` (clusters x resources, cluster-major), `economic_price`,
      #   `energy_price`, `latency` and `dependency` (clusters each),
      #   `replicas` (1) and `demand` (resources), and the total `size`.
      def observation_layout
        @observation_layout ||= begin
          n = clusters
          resources = RESOURCES.size
          Layout.new(free: 0, economic_price: n * resources, energy_price: (n * resources) + n,
            latency: (n * resources) + (2 * n), dependency: (n * resources) + (3 * n),
            replicas: (n * resources) + (4 * n), demand: (n * resources) + (4 * n) + 1,
            size: (n * resources) + (4 * n) + 1 + resources)
        end
      end

      def reset
        @step_count = 0
        @free = Array.new(clusters) { capacities.dup }
        @history = [@free.map(&:dup)]
        @economic_price = Array.new(clusters) { @random.rand }
        @energy_price = Array.new(clusters) { @random.rand }
        new_request
        observe
      end

      # @parameter action [Array(Numeric)] clusters + 1 values in [-1, 1].
      def step(action)
        raise ArgumentError, "expected #{clusters + 1} action components, got #{action.size}" unless action.size == clusters + 1

        replicas = @request[:replicas]
        counts = self.class.allocate(replicas, self.class.shares(action))
        outcome = deploy(counts)
        reward = reward_for(replicas, counts.last, outcome)

        recover
        @step_count += 1
        @history << @free.map(&:dup)
        @history.shift while @history.size > @config.max_delay + 1
        new_request
        done = @step_count >= @config.horizon

        metrics = {requested: replicas, served: outcome[:served], rejected: counts.last, overflow: outcome[:overflow],
                   cost: outcome[:cost], latency: outcome[:latency]}
        [observe, reward, done, {truncated: done, metrics: metrics}]
      end

      # The shares an action gives each cluster and the reject option.
      #
      # @parameter action [Array(Numeric)] values in [-1, 1].
      # @returns [Array(Float)] non-negative shares summing to 1.
      def self.shares(action)
        weights = action.map { |value| ((value + 1.0) / 2.0).clamp(0.0, 1.0) }
        total = weights.sum
        total.positive? ? weights.map { |weight| weight / total } : Array.new(action.size, 1.0 / action.size)
      end

      # Rounds `replicas * shares` to integer counts summing to `replicas`:
      # floors, then one more replica for the largest remainders.
      #
      # @returns [Array(Integer)]
      def self.allocate(replicas, shares)
        quotas = shares.map { |share| replicas * share }
        counts = quotas.map(&:floor)
        remainders = quotas.each_with_index.map { |quota, index| [quota - counts[index], index] }
        remainders.sort_by { |remainder, index| [-remainder, index] }.first(replicas - counts.sum).each { |_, index| counts[index] += 1 }
        counts
      end

      # @returns [Array(Float)] the capacity of each resource of a cluster.
      def capacities
        RESOURCES.map { |resource| @config.capacity.fetch(resource).to_f }
      end

      # What an observation says, in physical units.
      #
      # @parameter observation [Array(Float)] an observation of this environment.
      # @returns [Hash] `free` (the measured free resources, clusters x
      #   resources), `replicas`, `demand` (per replica, per resource),
      #   `latency` (normalized, per cluster) and `dependency` (a cluster
      #   index, or nil).
      def decode(observation)
        layout = observation_layout
        capacity = capacities
        n = clusters
        resources = RESOURCES.size
        max_demand = @config.max_demand_per_resource
        dependency = observation[layout.dependency, n].index { |value| value > 0.5 }
        {
          free: Array.new(n) { |cluster| Array.new(resources) { |resource| observation[layout.free + (cluster * resources) + resource] * capacity[resource] } },
          replicas: (observation[layout.replicas] * @config.replica_limit).round,
          demand: RESOURCES.each_with_index.map { |resource, index| (observation[layout.demand + index] * max_demand[resource]).round },
          latency: observation[layout.latency, n],
          dependency: dependency
        }
      end

      # The greedy best-fit heuristic: all the replicas go to the cluster
      # that can host them (according to the observation) leaving the least
      # free capacity; when none can, they are spread over the clusters with
      # the most free capacity, as many as fit in each, and the rest are
      # rejected.
      class GreedyPolicy
        include Reinforce::Agent

        def initialize(environment)
          @environment = environment
        end

        def predict(observation, deterministic: true)
          request = @environment.decode(observation)
          replicas = request[:replicas]
          capacity = @environment.capacities
          fits = request[:free].map { |free| fitting(free, request[:demand]) }
          counts = Array.new(@environment.clusters + 1, 0)

          hosts = fits.each_index.select { |cluster| fits[cluster] >= replicas }
          if hosts.any?
            best = hosts.min_by { |cluster| [leftover(request[:free][cluster], request[:demand], replicas, capacity), cluster] }
            counts[best] = replicas
          else
            remaining = replicas
            order = fits.each_index.sort_by { |cluster| [-leftover(request[:free][cluster], request[:demand], 0, capacity), cluster] }
            order.each do |cluster|
              placed = [fits[cluster], remaining].min
              counts[cluster] = placed
              remaining -= placed
            end
            counts[-1] = remaining
          end

          counts.map { |count| (2.0 * count / replicas) - 1.0 }
        end

        private

        def fitting(free, demand)
          free.each_with_index.filter_map { |amount, resource| (amount / demand[resource]).floor if demand[resource].positive? }.min
        end

        def leftover(free, demand, replicas, capacity)
          free.each_with_index.sum do |amount, resource|
            capacity[resource].positive? ? (amount - (replicas * demand[resource])) / capacity[resource] : 0.0
          end
        end
      end

      private

      def sample_latency_matrix
        n = clusters
        matrix = Array.new(n) { Array.new(n, 0.0) }
        n.times do |i|
          (i + 1...n).each { |j| matrix[i][j] = matrix[j][i] = @random.rand }
        end
        maximum = matrix.flatten.max
        maximum.positive? ? matrix.map { |row| row.map { |value| value / maximum } } : matrix
      end

      def new_request
        max_demand = @config.max_demand
        demand = {
          cpu: @random.rand(1..max_demand.fetch(:cpu)),
          memory: @random.rand(1..max_demand.fetch(:memory)),
          gpu: sample_gpu_demand,
          bandwidth: @random.rand(1..max_demand.fetch(:bandwidth))
        }
        dependency = (@random.rand < @config.dependency_probability) ? @random.rand(clusters) : nil
        @request = {replicas: @random.rand(1..@config.replica_limit), demand: demand, dependency: dependency}
        @measured = measure
      end

      def sample_gpu_demand
        threshold = @random.rand
        cumulative = 0.0
        @config.gpu_demand.each do |demand, probability|
          cumulative += probability
          return demand if threshold < cumulative
        end
        @config.gpu_demand.keys.last
      end

      # Each cluster's free resources as measured `delay` steps ago, the
      # delay drawn per cluster and resource; at the start of an episode,
      # delays beyond the history use the oldest measurement.
      def measure
        Array.new(clusters) do |cluster|
          RESOURCES.each_index.map do |resource|
            delay = @delay_random.rand(0..@config.max_delay)
            @history[[@history.size - 1 - delay, 0].max][cluster][resource]
          end
        end
      end

      def observe
        capacity = capacities
        free = @measured.flat_map do |cluster|
          cluster.each_with_index.map { |value, resource| capacity[resource].positive? ? value / capacity[resource] : 0.0 }
        end
        dependency = @request[:dependency]
        latency = Array.new(clusters) { |cluster| dependency ? @latency_matrix[cluster][dependency] : 0.0 }
        location = Array.new(clusters) { |cluster| (cluster == dependency) ? 1.0 : 0.0 }
        max_demand = @config.max_demand_per_resource
        demand = RESOURCES.map do |resource|
          max_demand[resource].positive? ? @request[:demand][resource].to_f / max_demand[resource] : 0.0
        end

        free + @economic_price + @energy_price + latency + location + [@request[:replicas].to_f / @config.replica_limit] + demand
      end

      def deploy(counts)
        demand = RESOURCES.map { |resource| @request[:demand][resource] }
        dependency = @request[:dependency]
        outcome = {served: 0, overflow: 0, cost: 0.0, latency: 0.0}
        clusters.times do |cluster|
          count = counts[cluster]
          next if count.zero?

          needed = demand.map { |amount| count * amount }
          if needed.each_with_index.all? { |amount, resource| amount <= @free[cluster][resource] }
            needed.each_with_index { |amount, resource| @free[cluster][resource] -= amount }
            outcome[:served] += count
            outcome[:cost] += count * (@economic_price[cluster] + @energy_price[cluster]) / 2.0
            outcome[:latency] += count * @latency_matrix[cluster][dependency] if dependency
          else
            fitting = demand.each_with_index.filter_map { |amount, resource| (@free[cluster][resource] / amount).floor if amount.positive? }.min
            outcome[:overflow] += count - fitting
          end
        end
        outcome
      end

      def reward_for(replicas, rejected, outcome)
        rejection_weight, overflow_weight, cost_weight, latency_weight = @config.reward_weights
        served = outcome[:served]
        cost = served.positive? ? outcome[:cost] / served : 0.0
        latency = served.positive? ? outcome[:latency] / served : 0.0
        1.0 - (rejection_weight * rejected / replicas) - (overflow_weight * outcome[:overflow] / replicas) -
          (cost_weight * cost) - (latency_weight * latency)
      end

      def recover
        capacity = capacities
        recovery = @config.recovery
        @free.each do |free|
          RESOURCES.each_with_index do |resource, index|
            amount = (resource == :gpu) ? @random.rand(0..recovery.fetch(:gpu)) : @random.rand * recovery.fetch(resource)
            free[index] = [free[index] + amount, capacity[index]].min
          end
        end
      end
    end
  end
end
