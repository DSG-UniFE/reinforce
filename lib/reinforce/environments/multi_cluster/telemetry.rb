# frozen_string_literal: true

module Reinforce
  module Environments
    class MultiCluster
      # A Prometheus-like monitoring pipeline, in the workload's time
      # (seconds for Workloads::Scripted; see docs/sensor-model.md):
      #
      # - each cluster is scraped every `interval`, at its own random phase,
      #   each scrape late by up to `jitter`; a scrape reads all of the
      #   cluster's resources at once;
      # - a reading becomes available after an ingestion latency, log-normal
      #   with median `latency_median` and log standard deviation
      #   `latency_sigma` (0 for none);
      # - a scrape fails with probability `failure`, leaving the previous
      #   reading in use;
      # - with `ledger`, the agent's own changes since a reading (what it
      #   deployed, minus its services that departed) correct it: the
      #   observation shows the *assumed* free amounts;
      # - the drift and volatility of the changes the agent did not make
      #   (other tenants) are estimated from consecutive readings by
      #   exponential moving averages with weight `smoothing`, from zero
      #   drift and a volatility of `prior_volatility` (a fraction of the
      #   capacity per square root of the time unit).
      #
      # Ages are normalized by `max_age` (nil: four intervals).
      Telemetry = Data.define(:interval, :jitter, :latency_median, :latency_sigma, :failure, :ledger, :max_age, :smoothing,
        :prior_volatility) do
        def initialize(interval: 15.0, jitter: 1.0, latency_median: 2.0, latency_sigma: 0.5, failure: 0.0, ledger: true, max_age: nil,
          smoothing: 0.1, prior_volatility: 0.02)
          super
        end

        # @returns [Float] the age that normalizes to 1.
        def age_scale
          max_age || (4.0 * interval)
        end

        # @returns [Monitor] the pipeline for one episode, which starts with
        #   an up-to-date reading of `free` at time 0.
        def start(free, capacity, random)
          Telemetry::Monitor.new(self, free, capacity, random)
        end
      end

      class Telemetry
        # What the agent knows about one cluster: the reading in use, when it
        # was scraped, the ledger since then, and the estimated drift and
        # volatility of other tenants' consumption (per resource, per time
        # unit and per square root of the time unit).
        Knowledge = Data.define(:reading, :scraped_at, :age, :ledger, :drift, :volatility)

        # @private
        class Monitor
          Reading = Data.define(:scraped_at, :available_at, :values)

          def initialize(telemetry, free, capacity, random)
            @telemetry = telemetry
            @capacity = capacity
            @random = random
            clusters = free.size
            initial = free.map { |values| Reading.new(scraped_at: 0.0, available_at: 0.0, values: values.dup) }
            @pending = Array.new(clusters) { [] } # scraped, not yet available, by availability
            @current = initial # the latest available reading, per cluster
            @own = Array.new(clusters) { [] } # [time, consumption per resource], by time
            # nominal scrape times: a random phase plus multiples of the interval
            @nominal = Array.new(clusters) { @random.rand * telemetry.interval }
            @next_scrape = @nominal.map { |time| time + jitter }
            @drift = Array.new(clusters) { Array.new(capacity.size, 0.0) }
            @variance = Array.new(clusters) { capacity.map { |amount| (telemetry.prior_volatility * amount)**2 } }
          end

          # Takes the scrapes due up to `time` (inclusive) of `free`, the
          # state just before any change at `time`.
          def advance(time, free)
            @next_scrape.each_index do |cluster|
              while @next_scrape[cluster] <= time
                scrape(cluster, @next_scrape[cluster], free[cluster])
                @nominal[cluster] += @telemetry.interval
                @next_scrape[cluster] = @nominal[cluster] + jitter
              end
            end
          end

          # Records a change the agent made at `time`: `consumption` (per
          # resource) taken on `cluster`, negative for its services' departures.
          def record_own(time, cluster, consumption)
            @own[cluster] << [time, consumption]
          end

          # @returns [Array(Knowledge)] per cluster, at `time`.
          def knowledge(time)
            @current.each_index.map do |cluster|
              refresh(cluster, time)
              reading = @current[cluster]
              Knowledge.new(reading: reading.values, scraped_at: reading.scraped_at, age: time - reading.scraped_at,
                ledger: own_since(cluster, reading.scraped_at), drift: @drift[cluster].dup,
                volatility: @variance[cluster].map { |variance| Math.sqrt(variance) })
            end
          end

          private

          def jitter
            @random.rand * @telemetry.jitter
          end

          def scrape(cluster, time, values)
            return if @random.rand < @telemetry.failure

            available = time + latency
            pending = @pending[cluster]
            index = pending.bsearch_index { |reading| reading.available_at > available } || pending.size
            pending.insert(index, Reading.new(scraped_at: time, available_at: available, values: values.dup))
          end

          def latency
            median = @telemetry.latency_median
            sigma = @telemetry.latency_sigma
            return median if sigma.zero? || median.zero?

            # Box-Muller
            normal = Math.sqrt(-2.0 * Math.log(1.0 - @random.rand)) * Math.cos(2.0 * Math::PI * @random.rand)
            median * Math.exp(sigma * normal)
          end

          # Makes the readings available by `time` current, in order of
          # scraping (an older reading arriving late is ignored), updating
          # the estimates of other tenants' consumption from each new one.
          def refresh(cluster, time)
            pending = @pending[cluster]
            while pending.any? && pending.first.available_at <= time
              reading = pending.shift
              previous = @current[cluster]
              next if reading.scraped_at <= previous.scraped_at

              estimate(cluster, previous, reading)
              @current[cluster] = reading
              # the ledger only needs the changes from the current reading on
              @own[cluster].shift while @own[cluster].any? && @own[cluster].first.first < reading.scraped_at
            end
          end

          # Between two readings, the other tenants' consumption grew by the
          # fall in free resources not explained by the agent's own changes.
          def estimate(cluster, previous, reading)
            elapsed = reading.scraped_at - previous.scraped_at
            return unless elapsed.positive?

            own = own_between(cluster, previous.scraped_at, reading.scraped_at)
            weight = @telemetry.smoothing
            @capacity.each_index do |resource|
              growth = previous.values[resource] - reading.values[resource] - own[resource]
              deviation = growth - (@drift[cluster][resource] * elapsed)
              @drift[cluster][resource] += weight * ((growth / elapsed) - @drift[cluster][resource])
              @variance[cluster][resource] += weight * ((deviation**2 / elapsed) - @variance[cluster][resource])
            end
          end

          # The agent's net consumption on `cluster` from `from` (inclusive) on.
          def own_since(cluster, from)
            totals = Array.new(@capacity.size, 0.0)
            @own[cluster].each do |time, consumption|
              next if time < from

              consumption.each_with_index { |amount, resource| totals[resource] += amount }
            end
            totals
          end

          # The agent's net consumption on `cluster` in [from, to).
          def own_between(cluster, from, to)
            totals = Array.new(@capacity.size, 0.0)
            @own[cluster].each do |time, consumption|
              next if time < from || time >= to

              consumption.each_with_index { |amount, resource| totals[resource] += amount }
            end
            totals
          end
        end
      end
    end
  end
end
