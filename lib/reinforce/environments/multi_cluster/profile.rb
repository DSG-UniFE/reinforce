# frozen_string_literal: true

module Reinforce
  module Environments
    class MultiCluster
      # A scriptable rate over time (e.g. arrivals per second): a sequence of
      # segments, with spikes added on top and an optional multiplicative
      # noise.
      #
      #   profile = Profile.build do
      #     constant 0.8, duration: 600
      #     ramp to: 1.6, duration: 300
      #     sine mean: 1.0, amplitude: 0.5, period: 1800, duration: 3600
      #     trace rows: [[0, 1.0], [60, 1.4], [120, 1.1]]
      #     spike peak: 4.0, at: 900, width: 60
      #     noise sigma: 0.2, correlation: 120
      #   end
      #
      # or, from data (e.g. YAML), Profile.from_spec([{"constant" => 0.8,
      # "duration" => 600}, {"ramp" => {"to" => 1.6, "duration" => 300}}]).
      #
      # Segments follow each other: `constant` holds a value, `ramp` moves
      # linearly from the previous segment's end value (or `from:`), `sine`
      # oscillates around a mean, and `trace` interpolates (time, rate) rows,
      # given inline or as a CSV file with a header. After the last segment
      # the profile holds its final value, or starts over if `cyclic`.
      # `spike` adds a Gaussian bump of height `peak` and standard deviation
      # `width` centred `at` a time; `noise` multiplies the rate by a
      # log-normal factor of mean 1 whose logarithm is an AR(1) process with
      # the given standard deviation and correlation time (capped at three
      # standard deviations, so the rate stays bounded).
      #
      # A profile is a value: #start gives the object that evaluates it for
      # one run (drawing its noise from a generator), whose #rate(time) never
      # exceeds #maximum.
      class Profile
        Segment = Data.define(:start, :duration, :maximum, :end_value, :function)
        Spike = Data.define(:peak, :at, :width)
        Noise = Data.define(:sigma, :correlation)

        attr_reader :segments, :spikes, :noise, :cyclic

        # @returns [Profile] a profile built by the block (see the class comment).
        def self.build(cyclic: false, &block)
          builder = Builder.new
          builder.instance_eval(&block)
          new(segments: builder.segments, spikes: builder.spikes, noise: builder.noise_spec, cyclic: cyclic)
        end

        # @returns [Profile] a constant rate.
        def self.constant(value)
          build { constant value, duration: 1.0 }
        end

        # @parameter spec [Array(Hash)] one Hash per segment, spike or noise,
        #   keyed by its name, e.g. {"sine" => {"mean" => 1, ...}}, or with
        #   the value inline and the options beside it, e.g.
        #   {"constant" => 0.8, "duration" => 600}.
        # @returns [Profile]
        def self.from_spec(spec, cyclic: false)
          build(cyclic: cyclic) do
            spec.each do |entry|
              entry = entry.transform_keys(&:to_sym)
              name = (entry.keys & Builder::NAMES).first or raise ArgumentError, "unknown profile entry: #{entry.inspect}"
              value = entry.delete(name)
              if value.is_a?(Hash)
                public_send(name, **value.transform_keys(&:to_sym), **entry)
              elsif value.nil?
                public_send(name, **entry)
              else
                public_send(name, value, **entry)
              end
            end
          end
        end

        def initialize(segments:, spikes: [], noise: nil, cyclic: false)
          raise ArgumentError, "a profile needs at least one segment" if segments.empty?

          @segments = segments.freeze
          @spikes = spikes.freeze
          @noise = noise
          @cyclic = cyclic
          freeze
        end

        # @returns [Float] the total duration of the segments.
        def duration
          @segments.last.start + @segments.last.duration
        end

        # @returns [Float] an upper bound of the rate.
        def maximum
          base = @segments.map(&:maximum).max + @spikes.sum(&:peak)
          @noise ? base * Math.exp(3.0 * @noise.sigma) : base
        end

        # @returns [Float] the rate at `time` without noise.
        def base_rate(time)
          time %= duration if @cyclic && duration.positive?
          segment = @segments.rfind { |candidate| candidate.start <= time } || @segments.first
          value = if time >= segment.start + segment.duration && segment.equal?(@segments.last)
            segment.end_value
          else
            segment.function.call(time - segment.start)
          end
          value += @spikes.sum { |spike| spike.peak * Math.exp(-0.5 * ((time - spike.at) / spike.width)**2) }
          [value, 0.0].max
        end

        # @parameter random [Random] for the noise.
        # @returns [Run] the profile's evaluation for one run.
        def start(random)
          Run.new(self, random)
        end

        # A profile's evaluation for one run, with its noise.
        class Run
          def initialize(profile, random)
            @profile = profile
            @random = random
            noise = profile.noise
            return unless noise

            @step = noise.correlation / 4.0
            @persistence = Math.exp(-@step / noise.correlation)
            @states = [gaussian]
          end

          def maximum = @profile.maximum

          # @returns [Float] the rate at `time` (>= 0).
          def rate(time)
            base = @profile.base_rate(time)
            noise = @profile.noise
            return base unless noise

            index = (time / @step).floor
            @states << ((@persistence * @states.last) + (Math.sqrt(1.0 - @persistence**2) * gaussian)) while @states.size <= index
            state = @states[index].clamp(-3.0, 3.0)
            base * Math.exp((noise.sigma * state) - (noise.sigma**2 / 2.0))
          end

          private

          # Box-Muller.
          def gaussian
            Math.sqrt(-2.0 * Math.log(1.0 - @random.rand)) * Math.cos(2.0 * Math::PI * @random.rand)
          end
        end

        # The methods available in Profile.build's block.
        class Builder
          NAMES = %i[constant ramp sine trace spike noise].freeze

          attr_reader :segments, :spikes, :noise_spec

          def initialize
            @segments = []
            @spikes = []
            @noise_spec = nil
          end

          def constant(value, duration:)
            value = value.to_f
            add(duration, value, value) { |_t| value }
          end

          def ramp(to:, duration:, from: nil)
            from = (from || @segments.last&.end_value || 0.0).to_f
            to = to.to_f
            duration = duration.to_f
            add(duration, [from, to].max, to) { |t| from + ((to - from) * (t / duration).clamp(0.0, 1.0)) }
          end

          def sine(mean:, amplitude:, period:, duration:, phase: 0.0)
            mean = mean.to_f
            amplitude = amplitude.to_f
            wave = ->(t) { mean + (amplitude * Math.sin((2.0 * Math::PI * t / period) + phase)) }
            add(duration, mean + amplitude.abs, wave.call(duration.to_f), &wave)
          end

          # @parameter path [String, nil] a CSV file with a header naming
          #   `time_column` (seconds from the segment's start) and `column`.
          # @parameter rows [Array(Array(Numeric)), nil] (time, rate) rows instead.
          def trace(path = nil, rows: nil, column: "rate", time_column: "time", scale: 1.0)
            rows ||= read_csv(path, time_column.to_s, column.to_s)
            rows = rows.map { |time, rate| [time.to_f, rate.to_f * scale] }.sort_by(&:first)
            raise ArgumentError, "a trace needs at least two rows" if rows.size < 2

            origin = rows.first.first
            rows = rows.map { |time, rate| [time - origin, rate] }
            times = rows.map(&:first)
            add(times.last, rows.map(&:last).max, rows.last.last) do |t|
              index = (times.bsearch_index { |time| time > t } || rows.size) - 1
              index = index.clamp(0, rows.size - 2)
              (t0, r0), (t1, r1) = rows[index], rows[index + 1]
              r0 + ((r1 - r0) * ((t - t0) / (t1 - t0)).clamp(0.0, 1.0))
            end
          end

          def spike(peak:, at:, width:)
            @spikes << Spike.new(peak: peak.to_f, at: at.to_f, width: width.to_f)
          end

          def noise(sigma:, correlation:)
            @noise_spec = Noise.new(sigma: sigma.to_f, correlation: correlation.to_f)
          end

          private

          def add(duration, maximum, end_value, &function)
            duration = duration.to_f
            raise ArgumentError, "a segment's duration must be positive" unless duration.positive?

            start = @segments.empty? ? 0.0 : @segments.last.start + @segments.last.duration
            @segments << Segment.new(start: start, duration: duration, maximum: maximum, end_value: end_value, function: function)
          end

          def read_csv(path, time_column, column)
            header, *lines = File.readlines(path, chomp: true).reject(&:empty?)
            names = header.split(",").map(&:strip)
            time_index = names.index(time_column) or raise ArgumentError, "#{path} has no #{time_column} column"
            rate_index = names.index(column) or raise ArgumentError, "#{path} has no #{column} column"
            lines.map do |line|
              fields = line.split(",")
              [Float(fields[time_index]), Float(fields[rate_index])]
            end
          end
        end
      end
    end
  end
end
