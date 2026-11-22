# frozen_string_literal: true

require_relative "profile"

module Reinforce
  module Environments
    class MultiCluster
      module Workloads
        # A change between two decisions, reported by a workload's #advance
        # to its block *before* it is applied (so that whatever watches the
        # clusters sees the state up to it): at `time`, `amounts` (per
        # resource) of `cluster` are released (`kind` :departure or
        # :release) or taken (:arrival); `own` tells whether they belong to
        # the agent's services.
        Event = Data.define(:time, :kind, :cluster, :amounts, :own)

        # Tenants other than the agent, on every cluster: services that
        # arrive (a Poisson process), are deployed on their cluster if they
        # fit (and dropped otherwise), and depart after a log-normal
        # lifetime. The agent sees them only through the clusters' free
        # resources.
        #
        # mean_load: the fraction of a cluster's capacity (for its most
        #   demanded resource) the tenants occupy on average, by Little's law;
        #   it sets their arrival rate.
        # profile: a Profile multiplying that rate over time (mean about 1);
        #   nil keeps it constant.
        # mean_lifetime, lifetime_sigma: of the log-normal lifetimes, in
        #   seconds and in log space.
        # replicas, max_demand, gpu_demand: the size of a service, as in a
        #   RequestClass; nil takes the environment's.
        Background = Data.define(:mean_load, :profile, :mean_lifetime, :lifetime_sigma, :replicas, :max_demand, :gpu_demand) do
          def initialize(mean_load: 0.3, profile: nil, mean_lifetime: 300.0, lifetime_sigma: 1.0, replicas: nil, max_demand: nil,
            gpu_demand: nil)
            super
          end

          # @returns [RequestClass] the tenants' services, filled in from the environment's Config.
          def service_class(config)
            Workloads.fill_classes([RequestClass.new(replicas: replicas, max_demand: max_demand, gpu_demand: gpu_demand)], config).first
          end

          # @returns [Float] arrivals per second per cluster at a profile multiplier of 1.
          def rate(config)
            occupation = Workloads.occupation_per_arrival([service_class(config)], config, clusters: 1)
            occupation.positive? ? mean_load / (mean_lifetime * occupation) : 0.0
          end
        end

        # A workload driven by scripts: agent requests arrive as a Poisson
        # process whose rate (per second) follows a Profile, and deployed
        # replicas hold their resources for a log-normal lifetime; optional
        # Background tenants consume resources on every cluster in between.
        # Time is in seconds, and each decision happens at an arrival.
        #
        # profile: the agent's arrival rate, a Profile.
        # classes: RequestClass mix (weights, replicas, demands, lifetime
        #   scales); nil is one class from the environment's Config.
        # mean_lifetime, lifetime_sigma: of the agent's services, in seconds
        #   and in log space.
        # background: Background tenants, or nil.
        Scripted = Data.define(:profile, :classes, :mean_lifetime, :lifetime_sigma, :background) do
          def initialize(profile: Profile.constant(1.0), classes: nil, mean_lifetime: 300.0, lifetime_sigma: 1.0, background: nil)
            super
          end

          def start(config, random)
            Scripted::Run.new(self, config, random)
          end

          def feature_size
            0
          end

          # Arrivals per second at the peak.
          def peak_rate
            profile.maximum
          end

          # The most a cluster gets back in a step: taken as one whole
          # request, as for Diurnal.
          def release_bound(config)
            config.max_demand_per_resource.transform_values { |maximum| (maximum * config.replica_limit).to_f }
          end
        end

        class Scripted
          # @private
          class Run
            # @returns [Float] the current time (seconds).
            attr_reader :time

            # @returns [Integer] background services dropped because they did not fit.
            attr_reader :background_dropped

            def initialize(workload, config, random)
              @workload = workload
              @config = config
              @random = random
              @classes = Workloads.fill_classes(workload.classes || [RequestClass.new], config)
              total = @classes.sum(&:weight)
              @weights = @classes.map { |request_class| [request_class, request_class.weight / total] }
              @profile = workload.profile.start(random)
              @occupation = Workloads.occupation_per_arrival(@classes, config, clusters: config.clusters)
              @departures = [] # [time, cluster, amounts, own], by time
              @time = 0.0
              @background_dropped = 0
              start_background(workload.background)
            end

            def next_request
              request_class = Workloads.sample_categorical(@random, @weights)
              demand = Workloads.sample_demand(@random, request_class.max_demand, request_class.gpu_demand)
              dependency = Workloads.sample_dependency(@random, @config)
              replicas = @random.rand(request_class.replicas)
              {replicas: replicas, demand: demand, dependency: dependency,
               lifetime: lifetime(@workload.mean_lifetime * request_class.lifetime_scale, @workload.lifetime_sigma)}
            end

            def deployed(cluster, amounts, request)
              schedule_departure(@time + request[:lifetime], cluster, amounts, true)
            end

            # Moves to the next agent arrival, applying the departures and
            # the background arrivals before it; each change is passed to
            # the block (an Event) before it is applied.
            def advance(free, capacity)
              target = next_arrival(@time)
              loop do
                departure = @departures.first&.first
                cluster = @background_next&.each_index&.min_by { |c| @background_next[c] }
                arrival = cluster && @background_next[cluster]
                break if [departure, arrival].compact.none? { |time| time <= target }

                if arrival && (departure.nil? || arrival < departure)
                  background_arrival(cluster, arrival, free) { |event| yield event if block_given? }
                else
                  time, cluster, amounts, own = @departures.shift
                  yield Event.new(time: time, kind: :departure, cluster: cluster, amounts: amounts, own: own) if block_given?
                  amounts.each_with_index { |amount, resource| free[cluster][resource] = [free[cluster][resource] + amount, capacity[resource]].min }
                end
              end
              @time = target
            end

            def features
              []
            end

            # @returns [Float] the load the agent's arrivals currently offer,
            #   as a fraction of the capacity of the most demanded resource.
            def offered_load
              @profile.rate(@time) * @workload.mean_lifetime * @occupation
            end

            private

            def start_background(background)
              return unless background

              @background = background
              @background_class = background.service_class(@config)
              @background_rate = background.rate(@config)
              @background_profile = (background.profile || Profile.constant(1.0)).start(@random)
              @background_next = Array.new(@config.clusters) { background_arrival_after(0.0) }
            end

            def background_arrival(cluster, time, free)
              @background_next[cluster] = background_arrival_after(time)
              demand = Workloads.sample_demand(@random, @background_class.max_demand, @background_class.gpu_demand)
              replicas = @random.rand(@background_class.replicas)
              amounts = RESOURCES.map { |resource| replicas * demand[resource].to_f }
              unless amounts.each_with_index.all? { |amount, resource| amount <= free[cluster][resource] }
                @background_dropped += 1
                return
              end

              yield Event.new(time: time, kind: :arrival, cluster: cluster, amounts: amounts, own: false)
              amounts.each_with_index { |amount, resource| free[cluster][resource] -= amount }
              schedule_departure(time + lifetime(@background.mean_lifetime, @background.lifetime_sigma), cluster, amounts, false)
            end

            def schedule_departure(time, cluster, amounts, own)
              index = @departures.bsearch_index { |other| other.first > time } || @departures.size
              @departures.insert(index, [time, cluster, amounts, own])
            end

            # The next agent arrival after `time`, by thinning (Lewis and
            # Shedler, 1979) against the profile's maximum.
            def next_arrival(time)
              thin(time, @profile)
            end

            def background_arrival_after(time)
              return Float::INFINITY unless @background_rate.positive?

              thin(time, @background_profile, scale: @background_rate)
            end

            def thin(time, profile, scale: 1.0)
              maximum = profile.maximum * scale
              raise ArgumentError, "a profile's maximum rate must be positive" unless maximum.positive?

              loop do
                time += -Math.log(1.0 - @random.rand) / maximum
                return time if @random.rand * maximum <= profile.rate(time) * scale
              end
            end

            def lifetime(mean, sigma)
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
