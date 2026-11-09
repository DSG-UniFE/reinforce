# frozen_string_literal: true

module Reinforce
  module Metrics
    # Collects the scalar metrics an algorithm produces while training
    # (episode rewards, losses, entropies, ...) and forwards each one to any
    # number of sinks as it is recorded.
    #
    # Every algorithm owns one, exposed as `#metrics` (and, for backward
    # compatibility, `#logs`). Reading works like the plain Hash of Arrays it
    # replaces -- `agent.metrics[:loss]` returns every recorded loss value --
    # so existing plotting and test code keeps working unchanged. Output
    # destinations are attached after construction:
    #
    #   agent.metrics.add_sink(Reinforce::Metrics::ConsoleSink.new(every: 50))
    #   agent.metrics.add_sink(Reinforce::Metrics::JSONLinesSink.new("run.jsonl"))
    #   agent.metrics.add_sink(->(key, value, step) { ... })
    #
    # A sink is anything responding to `call(key, value, step)`.
    #
    # Metric names used by every online algorithm: `episode_reward`,
    # `episode_length` (once per finished episode) and `loss` (once per
    # update). Algorithms record additional, algorithm-specific keys
    # alongside those (e.g. SAC's `q1_loss` or `alpha`).
    class Recorder
      include Enumerable

      attr_reader :sinks

      def initialize(sinks: [])
        @values = {}
        @sinks = sinks.dup
      end

      # Attach a sink; it only sees values recorded from now on.
      #
      # @parameter sink [#call] called as `sink.call(key, value, step)`.
      # @returns [self]
      def add_sink(sink)
        raise ArgumentError, "a metrics sink must respond to #call" unless sink.respond_to?(:call)

        @sinks << sink
        self
      end

      # Record one value of one metric.
      #
      # @parameter key [Symbol] the metric name.
      # @parameter value [Numeric] the value.
      # @parameter step [Integer, nil] the x-axis position of this value;
      #   defaults to the number of values previously recorded under `key`,
      #   i.e. its index in `self[key]`.
      # @returns [Numeric] the recorded value.
      def record(key, value, step: nil)
        series = (@values[key.to_sym] ||= [])
        step ||= series.size
        series << value
        @sinks.each { |sink| sink.call(key.to_sym, value, step) }
        value
      end

      # Record several metrics at once, sharing the same `step`.
      #
      # @parameter metrics [Hash{Symbol => Numeric}]
      # @returns [Hash{Symbol => Numeric}] the recorded metrics.
      def record_all(metrics, step: nil)
        metrics.each { |key, value| record(key, value, step: step) }
        metrics
      end

      # All values recorded under `key`, oldest first; empty if none were.
      # Returns a copy, so callers can't alter the recorded history.
      #
      # @returns [Array(Numeric)]
      def [](key)
        @values.fetch(key.to_sym, []).dup
      end

      # @returns [Array(Symbol)] the names of every metric recorded so far.
      def keys
        @values.keys
      end

      def key?(key)
        @values.key?(key.to_sym)
      end

      def each(&block)
        return enum_for(:each) unless block

        @values.each { |key, values| yield key, values.dup }
        self
      end

      # @returns [Numeric, nil] the most recently recorded value of `key`.
      def last(key)
        @values.fetch(key.to_sym, []).last
      end

      # The mean of the last `window` values of `key` (all of them when
      # `window` is nil), or nil if nothing was recorded.
      #
      # @returns [Float, nil]
      def mean(key, window: nil)
        values = @values.fetch(key.to_sym, [])
        values = values.last(window) if window
        return nil if values.empty?

        values.sum.to_f / values.size
      end

      # @returns [Hash{Symbol => Array(Numeric)}] a copy of every series.
      def to_h
        @values.transform_values(&:dup)
      end
    end
  end
end
