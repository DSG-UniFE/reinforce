# frozen_string_literal: true

require_relative "../environment"
require_relative "../spaces"
require_relative "multi_cluster/workloads"
require_relative "multi_cluster/telemetry"

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
    # - **Dynamics**: a workload (see Workloads) decides how requests
    #   arrive and how resources come back. By default (Turnover), every
    #   cluster recovers a random amount of each resource after each step,
    #   up to its capacity; with Diurnal (see .diurnal), deployed services
    #   hold their resources for a lifetime, under an arrival rate that
    #   rises and falls periodically.
    # - **Observations** (#observation_layout) are in [0, 1]: the measured
    #   free resources of each cluster (normalized by capacity), economic
    #   and energy prices, the normalized latency from each cluster to the
    #   request's dependency and its one-hot location, the normalized
    #   replica count and per-replica demands, then the workload's
    #   features (e.g. the phase of the cycle). Each measurement is delayed
    #   by a random number of steps in 0..max_delay, drawn independently
    #   per cluster and resource, or is the cluster's last scrape (see
    #   telemetry_interval). With `telemetry` (see Telemetry and
    #   docs/sensor-model.md), the free resources are those the agent
    #   assumes (its last reading of the cluster corrected by its own
    #   changes since), followed by each cluster's reading age, and the
    #   estimated drift and volatility of the other tenants' consumption.
    # - **Reward**: 1 - w1 * rejected/R - w2 * overflow/R - w3 * cost -
    #   w4 * latency, where cost and latency are averaged over the deployed
    #   replicas (prices and normalized latencies are in [0, 1]).
    #
    # Episodes last `horizon` steps and end truncated (`info[:truncated]`),
    # and `info[:metrics]` reports the step's requested, served, rejected
    # and overflowing replicas, the deployed replicas' total cost and
    # latency, the utilization of the most used resource, the background
    # services dropped (if the workload has background tenants) and, if the
    # workload has one, the offered load.
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
      # recovery: with the Turnover workload, the maximum amount of CPU,
      #   memory and bandwidth each cluster recovers per step (uniformly
      #   from 0), and of GPU (an integer, uniformly from 0).
      # max_delay: measurement delays are drawn uniformly in 0..max_delay
      #   steps, independently per cluster and resource.
      # telemetry_interval: if set, monitoring instead scrapes each cluster
      #   (all its resources at once) every telemetry_interval time units of
      #   the workload, each cluster at its own random phase, and an
      #   observation shows the last scrape. Measurements are then as old in
      #   time at peaks as at troughs, but span more decisions at peaks.
      # dependency_probability: the probability that a request depends on a
      #   service hosted by a (uniformly chosen) cluster.
      # horizon: the length of an episode.
      # reward_weights: the weights of rejection, overflow, cost and latency.
      # workload: how requests arrive and resources are released (see
      #   Workloads); nil means Workloads::Turnover.
      # telemetry: a Telemetry pipeline, replacing max_delay and
      #   telemetry_interval; nil keeps them.
      Config = Data.define(:clusters, :capacity, :max_demand, :gpu_demand, :max_replicas, :recovery, :max_delay,
        :telemetry_interval, :dependency_probability, :horizon, :reward_weights, :workload, :telemetry) do
        def initialize(clusters: 4, capacity: {cpu: 50, memory: 100, gpu: 2, bandwidth: 1000},
          max_demand: {cpu: 5, memory: 10, bandwidth: 100}, gpu_demand: {0 => 0.8, 1 => 0.2}, max_replicas: nil,
          recovery: {cpu: 25, memory: 50, gpu: 1, bandwidth: 500}, max_delay: 1, telemetry_interval: nil, dependency_probability: 0.5,
          horizon: 1_000, reward_weights: [2.0, 2.0, 0.5, 0.5], workload: nil, telemetry: nil)
          raise ArgumentError, "telemetry replaces telemetry_interval: set one of them" if telemetry && telemetry_interval

          super
        end

        # @returns [Integer] the maximum replica count of a request.
        def replica_limit
          max_replicas || (5 * clusters / 2)
        end

        # @returns [#start] the workload, Workloads::Turnover by default.
        def workload_model
          workload || Workloads::Turnover.new
        end

        # @returns [Hash{Symbol => Numeric}] the maximum per-replica demand of every resource.
        def max_demand_per_resource
          max_demand.merge(gpu: gpu_demand.keys.max)
        end
      end

      # Offsets of the observation's components (see #observation_layout).
      # `age`, `drift` and `volatility` are nil without telemetry.
      Layout = Data.define(:free, :economic_price, :energy_price, :latency, :dependency, :replicas, :demand, :workload, :age, :drift,
        :volatility, :size)

      # Small requests and fast recovery.
      def self.easy(**options)
        new(**options)
      end

      # Larger requests and slower recovery of CPU and bandwidth, so the
      # infrastructure cannot serve the whole workload at steady state.
      def self.hard(**options)
        new(max_demand: {cpu: 10, memory: 20, bandwidth: 200}, recovery: {cpu: 10, memory: 100, gpu: 1, bandwidth: 200}, **options)
      end

      # Services with lifetimes under a periodic arrival rate (see
      # Workloads::Diurnal, whose parameters `workload` overrides), on
      # clusters four times larger than .easy's, so that each holds tens of
      # services rather than a few and the load, not the size of single
      # requests, decides what fits; monitoring scrapes every 5 time units
      # (mean interarrival times).
      #
      # By default the offered load swings between 0.5 and 1.2 of the
      # capacity over a period of 600 arrivals, with services living about
      # 25 arrivals on average.
      def self.diurnal(workload: {}, **options)
        workload = Workloads::Diurnal.new(period: 600.0, **workload)
        new(capacity: {cpu: 200, memory: 400, gpu: 8, bandwidth: 4000}, telemetry_interval: 5.0, workload: workload, **options)
      end

      attr_reader :config, :latency_matrix, :free, :request, :workload

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
      #   `replicas` (1), `demand` (resources), the workload's features,
      #   with telemetry the reading `age` (clusters), the other tenants'
      #   `drift` and `volatility` (clusters x resources each), and the
      #   total `size`.
      def observation_layout
        @observation_layout ||= begin
          n = clusters
          resources = RESOURCES.size
          workload = (n * resources) + (4 * n) + 1 + resources
          size = workload + @config.workload_model.feature_size
          age = drift = volatility = nil
          if @config.telemetry
            age = size
            drift = age + n
            volatility = drift + (n * resources)
            size = volatility + (n * resources)
          end
          Layout.new(free: 0, economic_price: n * resources, energy_price: (n * resources) + n,
            latency: (n * resources) + (2 * n), dependency: (n * resources) + (3 * n),
            replicas: (n * resources) + (4 * n), demand: (n * resources) + (4 * n) + 1,
            workload: workload, age: age, drift: drift, volatility: volatility, size: size)
        end
      end

      def reset
        @step_count = 0
        @free = Array.new(clusters) { capacities.dup }
        @workload = @config.workload_model.start(@config, @random)
        @history = [[@workload.time, @free.map(&:dup)]]
        interval = @config.telemetry_interval
        @scrape_phase = Array.new(clusters) { interval ? @delay_random.rand * interval : 0.0 }
        @economic_price = Array.new(clusters) { @random.rand }
        @energy_price = Array.new(clusters) { @random.rand }
        @monitor = @config.telemetry&.start(@free, capacities, @delay_random)
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
        dropped = background_dropped

        if @monitor
          outcome[:deployments].each do |deployment|
            @monitor.record_own(@workload.time, deployment[:cluster], deployment[:amounts]) if deployment[:fitted]
          end
          @workload.advance(@free, capacities) do |event|
            @monitor.advance(event.time, @free)
            @monitor.record_own(event.time, event.cluster, event.amounts.map(&:-@)) if event.own
          end
          @monitor.advance(@workload.time, @free)
        else
          @workload.advance(@free, capacities)
        end
        @step_count += 1
        record_history
        new_request
        done = @step_count >= @config.horizon

        metrics = {requested: replicas, served: outcome[:served], rejected: counts.last, overflow: outcome[:overflow],
                   cost: outcome[:cost], latency: outcome[:latency], utilization: utilization}
        offered_load = @workload.offered_load
        metrics[:offered_load] = offered_load if offered_load
        metrics[:background_dropped] = background_dropped - dropped if dropped
        deployments = outcome[:deployments].map { |deployment| deployment.slice(:cluster, :replicas, :fitted) }
        [observe, reward, done, {truncated: done, metrics: metrics, deployments: deployments}]
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

      # The most a cluster's free resources can grow in a step, per
      # resource, according to the workload (used to bound how stale
      # measurements can be wrong).
      #
      # @returns [Hash{Symbol => Float}]
      def release_bound
        @config.workload_model.release_bound(@config)
      end

      # The most steps a measurement can be behind: max_delay, or with
      # telemetry_interval the arrivals expected in one interval at the
      # workload's peak rate.
      #
      # @returns [Integer]
      def telemetry_delay_bound
        interval = @config.telemetry_interval
        interval ? (interval * @config.workload_model.peak_rate).ceil : @config.max_delay
      end

      # @returns [Float] the fraction of the capacity in use, for the most
      #   used resource (over all the clusters).
      def utilization
        capacity = capacities
        RESOURCES.each_index.filter_map do |resource|
          total = capacity[resource] * clusters
          1.0 - (@free.sum { |cluster| cluster[resource] } / total) if total.positive?
        end.max
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
      # rejected. With `oracle: true` it reads the clusters' actual free
      # resources instead of the (possibly stale) measurements, as a
      # reference for what perfect telemetry is worth.
      class GreedyPolicy
        include Reinforce::Agent

        def initialize(environment, oracle: false)
          @environment = environment
          @oracle = oracle
        end

        def predict(observation, deterministic: true)
          request = @environment.decode(observation)
          request[:free] = @environment.free.map(&:dup) if @oracle
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
        @request = @workload.next_request
        @measured = @monitor ? monitored : measure
      end

      # @returns [Integer, nil] the background services dropped so far, if the workload has any.
      def background_dropped
        @workload.respond_to?(:background_dropped) ? @workload.background_dropped : nil
      end

      # With telemetry: each cluster's free resources as the agent assumes
      # them (its reading corrected by its own changes since, within the
      # capacity), or as read without the ledger; keeps what it knows for
      # the observation.
      def monitored
        capacity = capacities
        @knowledge = @monitor.knowledge(@workload.time)
        @knowledge.map do |known|
          next known.reading.dup unless @config.telemetry.ledger

          known.reading.each_with_index.map { |value, resource| (value - known.ledger[resource]).clamp(0.0, capacity[resource]) }
        end
      end

      # The observation's telemetry components (see #observation_layout).
      def telemetry_features
        capacity = capacities
        scale = @config.telemetry.age_scale
        ages = @knowledge.map { |known| (known.age / scale).clamp(0.0, 1.0) }
        drift = @knowledge.flat_map do |known|
          known.drift.each_with_index.map { |rate, resource| capacity[resource].positive? ? (((rate * scale / capacity[resource]).clamp(-1.0, 1.0) + 1.0) / 2.0) : 0.5 }
        end
        volatility = @knowledge.flat_map do |known|
          known.volatility.each_with_index.map { |sigma, resource| capacity[resource].positive? ? (sigma * Math.sqrt(scale) / capacity[resource]).clamp(0.0, 1.0) : 0.0 }
        end
        ages + drift + volatility
      end

      # Each cluster's free resources as measured `delay` steps ago, the
      # delay drawn per cluster and resource; at the start of an episode,
      # delays beyond the history use the oldest measurement.
      def measure
        return scraped if @config.telemetry_interval

        Array.new(clusters) do |cluster|
          RESOURCES.each_index.map do |resource|
            delay = @delay_random.rand(0..@config.max_delay)
            @history[[@history.size - 1 - delay, 0].max][1][cluster][resource]
          end
        end
      end

      # Each cluster's free resources at its last scrape (the state after
      # the last step before it).
      def scraped
        now = @workload.time
        interval = @config.telemetry_interval
        Array.new(clusters) do |cluster|
          scrape = now - ((now - @scrape_phase[cluster]) % interval)
          entry = @history.rfind { |time, _free| time <= scrape } || @history.first
          entry[1][cluster].dup
        end
      end

      def record_history
        @history << [@workload.time, @free.map(&:dup)]
        if (interval = @config.telemetry_interval)
          # keep the state in effect at the oldest possible scrape
          horizon = @workload.time - interval
          @history.shift while @history.size > 1 && @history[1][0] <= horizon
        else
          @history.shift while @history.size > @config.max_delay + 1
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

        observation = free + @economic_price + @energy_price + latency + location + [@request[:replicas].to_f / @config.replica_limit] + demand +
          @workload.features
        @monitor ? observation + telemetry_features : observation
      end

      def deploy(counts)
        demand = RESOURCES.map { |resource| @request[:demand][resource] }
        dependency = @request[:dependency]
        outcome = {served: 0, overflow: 0, cost: 0.0, latency: 0.0, deployments: []}
        clusters.times do |cluster|
          count = counts[cluster]
          next if count.zero?

          needed = demand.map { |amount| count * amount }
          fitted = needed.each_with_index.all? { |amount, resource| amount <= @free[cluster][resource] }
          outcome[:deployments] << {cluster: cluster, replicas: count, fitted: fitted, amounts: needed.map(&:to_f)}
          if fitted
            needed.each_with_index { |amount, resource| @free[cluster][resource] -= amount }
            outcome[:served] += count
            @workload.deployed(cluster, needed, @request)
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
    end
  end
end
