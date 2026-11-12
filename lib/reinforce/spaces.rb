# frozen_string_literal: true

module Reinforce
  # Action spaces that environments can declare explicitly (see
  # Reinforce::Environment#action_space).
  module Spaces
    # A continuous space: vectors whose components lie between `low` and
    # `high`, inclusive.
    #
    #   space = Reinforce::Spaces::Box.new(low: [-1.0, -1.0], high: [1.0, 1.0])
    #   space.sample # => e.g. [0.31, -0.72]
    #
    # Continuous-control algorithms (e.g. Reinforce::Algorithms::ContinuousSAC)
    # act in [-1, 1] on each component and use #scale to map their actions
    # onto the space.
    class Box
      # @attribute [Array(Float)] the lower bound of each component.
      attr_reader :low

      # @attribute [Array(Float)] the upper bound of each component.
      attr_reader :high

      # @parameter low [Array(Numeric), Numeric] the lower bounds, or one bound for every component.
      # @parameter high [Array(Numeric), Numeric] the upper bounds, or one bound for every component.
      # @parameter size [Integer, nil] the number of components, needed when both bounds are numbers.
      # @raises [ArgumentError] if the bounds have different sizes or low > high.
      def initialize(low:, high:, size: nil)
        size ||= [low, high].find { |bound| bound.is_a?(Array) }&.size
        raise ArgumentError, "size is required when low and high are both numbers" unless size

        @low = expand(low, size)
        @high = expand(high, size)
        raise ArgumentError, "low and high must have #{size} components" unless @low.size == size && @high.size == size
        raise ArgumentError, "low must not exceed high" if @low.zip(@high).any? { |l, h| l > h }

        freeze
      end

      # @returns [Integer] the number of components.
      def size
        @low.size
      end

      # @returns [Array(Float)] a uniformly random element of the space.
      def sample
        @low.zip(@high).map { |l, h| l + (rand * (h - l)) }
      end

      # @returns [Boolean] whether `action` belongs to the space.
      def contains?(action)
        action.size == size && action.each_with_index.all? { |value, index| value.between?(@low[index], @high[index]) }
      end

      # Maps a vector from [-1, 1] on each component onto the space.
      #
      # @parameter unit [Array(Numeric)] values in [-1, 1].
      # @returns [Array(Float)]
      def scale(unit)
        unit.each_with_index.map { |value, index| @low[index] + ((value + 1.0) * 0.5 * (@high[index] - @low[index])) }
      end

      # The inverse of #scale.
      #
      # @parameter action [Array(Numeric)] an element of the space.
      # @returns [Array(Float)] values in [-1, 1].
      def unscale(action)
        action.each_with_index.map do |value, index|
          width = @high[index] - @low[index]
          width.zero? ? 0.0 : ((2.0 * (value - @low[index]) / width) - 1.0)
        end
      end

      def ==(other)
        other.is_a?(Box) && other.low == @low && other.high == @high
      end
      alias_method :eql?, :==

      def hash
        [self.class, @low, @high].hash
      end

      def inspect
        "Box(low: #{@low.inspect}, high: #{@high.inspect})"
      end

      private

      def expand(bound, size)
        bound.is_a?(Array) ? bound.map(&:to_f) : Array.new(size, bound.to_f)
      end
    end
  end
end
