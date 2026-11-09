# frozen_string_literal: true

module Reinforce
  module Metrics
    # Prints a one-line summary of training progress every `every` values of
    # the `trigger` metric (by default, every `every` finished episodes).
    # Each line shows, for every metric recorded since the previous line,
    # the mean of the values recorded in between:
    #
    #   [episode_reward 50] episode_reward=-12.4 episode_length=18.2 loss=0.0311
    #
    # Metrics are listed in the order they were first seen, trigger first.
    class ConsoleSink
      # @parameter output [IO] where to print (`$stdout` by default).
      # @parameter every [Integer] print once per this many trigger values.
      # @parameter trigger [Symbol] the metric whose count drives printing.
      # @parameter precision [Integer] significant digits shown per value.
      def initialize(output: $stdout, every: 10, trigger: :episode_reward, precision: 4)
        raise ArgumentError, "every must be a positive Integer" unless every.is_a?(Integer) && every.positive?

        @output = output
        @every = every
        @trigger = trigger.to_sym
        @precision = precision
        @trigger_count = 0
        @pending = {}
      end

      def call(key, value, _step)
        (@pending[key] ||= []) << value
        return unless key == @trigger

        @trigger_count += 1
        flush if (@trigger_count % @every).zero?
      end

      private

      def flush
        keys = [@trigger] + (@pending.keys - [@trigger])
        fields = keys.map { |key| "#{key}=#{format_value(mean(@pending[key]))}" }
        @output.puts "[#{@trigger} #{@trigger_count}] #{fields.join(" ")}"
        @pending.clear
      end

      def mean(values)
        values.sum.to_f / values.size
      end

      def format_value(value)
        format("%.#{@precision}g", value)
      end
    end
  end
end
