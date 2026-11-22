# frozen_string_literal: true

module Reinforce
  module Environments
    class MultiCluster
      # How requests arrive and how deployed services give their resources
      # back. A workload is a value (its parameters) whose #start returns the
      # object that runs it for one episode:
      #
      # - `next_request` returns the next request: `replicas`, per-replica
      #   `demand` (by resource), `dependency` (a cluster index, or nil) and
      #   whatever the workload needs about it (e.g. its `lifetime`);
      # - `deployed(cluster, amounts, request)` records that `amounts` of
      #   `cluster`'s resources now serve `request`;
      # - `advance(free, capacity)` moves to the next arrival, giving
      #   resources back to `free` (clusters x resources, in place), and
      #   passes each change to its block, if given, as an Event before
      #   applying it;
      # - `features` are extra observation components, in [0, 1];
      # - `offered_load` is the load the arrivals currently offer, as a
      #   fraction of the capacity (nil if the workload has no notion of it).
      module Workloads
        # Samples a request's per-replica demand: CPU, memory and bandwidth
        # uniformly in 1..max_demand, and GPU from its distribution.
        # @private
        def self.sample_demand(random, max_demand, gpu_demand)
          {
            cpu: random.rand(1..max_demand.fetch(:cpu)),
            memory: random.rand(1..max_demand.fetch(:memory)),
            gpu: sample_categorical(random, gpu_demand),
            bandwidth: random.rand(1..max_demand.fetch(:bandwidth))
          }
        end

        # @private
        def self.sample_categorical(random, distribution)
          threshold = random.rand
          cumulative = 0.0
          distribution.each do |value, probability|
            cumulative += probability
            return value if threshold < cumulative
          end
          distribution.keys.last
        end

        # @private
        def self.sample_dependency(random, config)
          (random.rand < config.dependency_probability) ? random.rand(config.clusters) : nil
        end

        # Request classes with what they leave nil filled in from the
        # environment's Config, and checked against its limits (which
        # normalize the observations).
        # @private
        def self.fill_classes(classes, config)
          classes.map do |request_class|
            filled = request_class.with(
              replicas: request_class.replicas || (1..config.replica_limit),
              max_demand: request_class.max_demand || config.max_demand,
              gpu_demand: request_class.gpu_demand || config.gpu_demand
            )
            if filled.replicas.last > config.replica_limit || filled.max_demand.any? { |resource, maximum| maximum > config.max_demand.fetch(resource) } ||
                filled.gpu_demand.keys.max > config.gpu_demand.keys.max
              raise ArgumentError, "request classes must stay within the environment's replica and demand limits"
            end
            filled
          end
        end

        # The fraction of the capacity of `clusters` clusters (for the most
        # demanded resource) that one arrival per time unit occupies per
        # unit of mean lifetime, with the classes' mix and lifetime scales:
        # by Little's law, the occupation is rate x mean lifetime x this.
        # @private
        def self.occupation_per_arrival(classes, config, clusters:)
          total_weight = classes.sum(&:weight)
          capacity = config.capacity
          RESOURCES.filter_map do |resource|
            total = capacity.fetch(resource) * clusters
            next unless total.positive?

            classes.sum { |request_class| request_class.weight * request_class.lifetime_scale * request_class.expected_demand(resource) } /
              total_weight / total
          end.max
        end

        # One request per step, and each cluster recovers a random amount of
        # every resource per step (`recovery` in the environment's Config),
        # up to its capacity, whatever is running on it. Simple, but the
        # amount recovered does not depend on what was deployed, and the load
        # is stationary.
        Turnover = Data.define do
          def start(config, random)
            Turnover::Run.new(config, random)
          end

          def feature_size
            0
          end

          # Arrivals per time unit at the peak: one per step.
          def peak_rate
            1.0
          end

          # @returns [Hash{Symbol => Float}] the most a cluster recovers per step.
          def release_bound(config)
            config.recovery.transform_values(&:to_f)
          end
        end

        class Turnover
          # @private
          class Run
            def initialize(config, random)
              @config = config
              @random = random
              @time = 0
            end

            # @returns [Integer] the steps taken (one per time unit).
            attr_reader :time

            def next_request
              demand = Workloads.sample_demand(@random, @config.max_demand, @config.gpu_demand)
              dependency = Workloads.sample_dependency(@random, @config)
              {replicas: @random.rand(1..@config.replica_limit), demand: demand, dependency: dependency}
            end

            def deployed(_cluster, _amounts, _request)
            end

            def advance(free, capacity)
              @time += 1
              recovery = @config.recovery
              free.each_with_index do |cluster, cluster_index|
                amounts = RESOURCES.map do |resource|
                  (resource == :gpu) ? @random.rand(0..recovery.fetch(:gpu)) : @random.rand * recovery.fetch(resource)
                end
                yield Event.new(time: @time, kind: :release, cluster: cluster_index, amounts: amounts, own: false) if block_given?
                amounts.each_with_index { |amount, index| cluster[index] = [cluster[index] + amount, capacity[index]].min }
              end
            end

            def features
              []
            end

            def offered_load
              nil
            end
          end
        end

        # A class of requests. `replicas` is a Range; `max_demand` gives
        # the maximum per-replica CPU, memory and bandwidth (drawn from 1);
        # `gpu_demand` is {demand => probability}.
        RequestClass = Data.define(:weight, :replicas, :max_demand, :gpu_demand, :lifetime_scale) do
          def initialize(weight: 1.0, replicas: nil, max_demand: nil, gpu_demand: nil, lifetime_scale: 1.0)
            super
          end

          # The expected total demand of a request, per resource.
          def expected_demand(resource)
            mean_replicas = (replicas.first + replicas.last) / 2.0
            per_replica = if resource == :gpu
              gpu_demand.sum { |demand, probability| demand * probability }
            else
              (1 + max_demand.fetch(resource)) / 2.0
            end
            mean_replicas * per_replica
          end
        end

        # Services that hold their resources for a lifetime, under an arrival
        # rate that rises and falls periodically, like a daily cycle.
        #
        # - **Arrivals** form a Poisson process whose rate varies as
        #   1 + amplitude * sin(2 pi t / period + phase) (simulated by
        #   thinning); time is measured in mean interarrival times, so a
        #   period of 200 holds 200 requests on average. Each step is one
        #   arrival: at the peaks, decisions are closer together in time.
        # - **Lifetimes**: the replicas a cluster deploys for a request keep
        #   their resources for the request's lifetime (log-normal, with
        #   log standard deviation `lifetime_sigma`), then give exactly those
        #   resources back. Nothing else is released, so what an agent
        #   deploys shapes the capacity available later.
        # - **Load**: `mean_load` is the fraction of the capacity that the
        #   arrivals occupy on average (by Little's law, arrival rate x mean
        #   demand x mean lifetime), for the most demanded resource; the mean
        #   lifetime is set to achieve it. The offered load then swings
        #   between about mean_load * (1 - amplitude) and
        #   mean_load * (1 + amplitude); with mean_load 0.85 and amplitude
        #   0.4, from 0.5 (idle clusters) to 1.2 (overload).
        # - **Classes**: requests belong to classes, with a weight (their
        #   share of the arrivals), replica and demand ranges, a GPU demand
        #   distribution and a lifetime scale (relative to the mean). By
        #   default there is one class, from the environment's Config.
        #
        # The period should be much longer than the mean lifetime (see
        # #mean_lifetime), so that occupation follows the arrival rate.
        Diurnal = Data.define(:mean_load, :amplitude, :period, :phase, :lifetime_sigma, :classes, :observe_phase) do
          # @parameter observe_phase [Boolean] whether observations include
          #   the phase of the cycle (its sine and cosine, scaled to [0, 1]),
          #   as an operator knows the time of day.
          def initialize(mean_load: 0.85, amplitude: 0.4, period: 200.0, phase: 0.0, lifetime_sigma: 1.0, classes: nil,
            observe_phase: true)
            raise ArgumentError, "amplitude must be in [0, 1)" unless amplitude >= 0.0 && amplitude < 1.0

            super
          end

          def start(config, random)
            Diurnal::Run.new(self, config, random)
          end

          def feature_size
            observe_phase ? 2 : 0
          end

          # Arrivals per time unit at the peak.
          def peak_rate
            1.0 + amplitude
          end

          # The request classes, filled in from the environment's Config
          # where left nil, and checked against its limits (which normalize
          # the observations).
          def request_classes(config)
            Workloads.fill_classes(classes || [RequestClass.new], config)
          end

          # The mean lifetime that makes the arrivals occupy `mean_load` of
          # the capacity of the most demanded resource (Little's law, with one
          # arrival per time unit on average).
          def mean_lifetime(config)
            mean_load / Workloads.occupation_per_arrival(request_classes(config), config, clusters: config.clusters)
          end

          # The most a cluster gets back in a step: taken as one whole request,
          # as many departures as arrivals happen on average.
          def release_bound(config)
            config.max_demand_per_resource.transform_values { |maximum| (maximum * config.replica_limit).to_f }
          end
        end

        class Diurnal
          # @private
          class Run
            def initialize(workload, config, random)
              @workload = workload
              @config = config
              @random = random
              @classes = workload.request_classes(config)
              total = @classes.sum(&:weight)
              @weights = @classes.map { |request_class| [request_class, request_class.weight / total] }
              @mean_lifetime = workload.mean_lifetime(config)
              @time = 0.0
              @departures = []
            end

            attr_reader :time

            def next_request
              request_class = Workloads.sample_categorical(@random, @weights)
              demand = Workloads.sample_demand(@random, request_class.max_demand, request_class.gpu_demand)
              dependency = Workloads.sample_dependency(@random, @config)
              replicas = @random.rand(request_class.replicas)
              {replicas: replicas, demand: demand, dependency: dependency, lifetime: lifetime(request_class)}
            end

            def deployed(cluster, amounts, request)
              departure = [@time + request[:lifetime], cluster, amounts]
              index = @departures.bsearch_index { |other| other.first > departure.first } || @departures.size
              @departures.insert(index, departure)
            end

            def advance(free, capacity)
              @time = next_arrival
              while @departures.any? && @departures.first.first <= @time
                time, cluster, amounts = @departures.shift
                yield Event.new(time: time, kind: :departure, cluster: cluster, amounts: amounts, own: true) if block_given?
                amounts.each_with_index { |amount, resource| free[cluster][resource] = [free[cluster][resource] + amount, capacity[resource]].min }
              end
            end

            def features
              return [] unless @workload.observe_phase

              angle = angle_at(@time)
              [(Math.sin(angle) + 1.0) / 2.0, (Math.cos(angle) + 1.0) / 2.0]
            end

            # @returns [Float] the offered load at the current time.
            def offered_load
              @workload.mean_load * rate(@time)
            end

            # @returns [Integer] the deployments still holding resources.
            def active
              @departures.size
            end

            private

            def angle_at(time)
              (2.0 * Math::PI * time / @workload.period) + @workload.phase
            end

            # The arrival rate, relative to its mean.
            def rate(time)
              1.0 + (@workload.amplitude * Math.sin(angle_at(time)))
            end

            # Thinning (Lewis and Shedler, 1979): candidates at the maximum
            # rate, each kept with probability rate / maximum rate.
            def next_arrival
              maximum = 1.0 + @workload.amplitude
              time = @time
              loop do
                time += -Math.log(1.0 - @random.rand) / maximum
                return time if @random.rand * maximum <= rate(time)
              end
            end

            def lifetime(request_class)
              mean = @mean_lifetime * request_class.lifetime_scale
              sigma = @workload.lifetime_sigma
              mu = Math.log(mean) - (sigma**2 / 2.0)
              # Box-Muller
              normal = Math.sqrt(-2.0 * Math.log(1.0 - @random.rand)) * Math.cos(2.0 * Math::PI * @random.rand)
              Math.exp(mu + (sigma * normal))
            end
          end
        end
      end
    end
  end
end

require_relative "scripted"
