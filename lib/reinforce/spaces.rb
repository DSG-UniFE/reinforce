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

    # A multi-discrete space: vectors of indices, the i-th in 0...sizes[i]
    # (one categorical choice per dimension, e.g. scale each service up,
    # down or not at all).
    #
    #   space = Reinforce::Spaces::MultiDiscrete.new([3, 3, 2])
    #   space.sample # => e.g. [2, 0, 1]
    #
    # An environment declaring one (see Reinforce::Environment#action_space)
    # takes such vectors in #step, and its #action_mask, if any, has one
    # Array of booleans per dimension.
    class MultiDiscrete
      # @attribute [Array(Integer)] the number of choices in each dimension.
      attr_reader :sizes

      def initialize(sizes)
        raise ArgumentError, "sizes must be positive Integers" if sizes.empty? || !sizes.all? { |n| n.is_a?(Integer) && n.positive? }

        @sizes = sizes.dup.freeze
        freeze
      end

      # @returns [Integer] the number of dimensions.
      def size = @sizes.size

      # @returns [Integer] the number of choices over all dimensions (the
      #   width of a policy head producing one logit per choice).
      def total = @sizes.sum

      def sample = @sizes.map { |n| rand(n) }

      def contains?(action)
        action.size == size && action.each_with_index.all? { |index, i| index.is_a?(Integer) && index.between?(0, @sizes[i] - 1) }
      end

      def ==(other) = other.is_a?(MultiDiscrete) && other.sizes == @sizes
      alias_method :eql?, :==

      def hash = [self.class, @sizes].hash

      def inspect = "MultiDiscrete(#{@sizes.inspect})"
    end

    # The environment's multi-discrete action space, or nil when its
    # actions are discrete (it lists them in #actions).
    def self.multi_discrete(environment)
      return nil unless environment.respond_to?(:action_space)

      space = begin
        environment.action_space
      rescue NotImplementedError
        nil
      end
      space.is_a?(MultiDiscrete) ? space : nil
    end

    # The environment's current action mask as one flat Array of booleans
    # (dimensions concatenated for a multi-discrete space), or nil when it
    # doesn't mask actions.
    def self.flat_action_mask(environment)
      return nil unless environment.respond_to?(:action_mask)

      mask = environment.action_mask
      return nil if mask.nil?

      flat = mask.first.is_a?(Array) ? mask.flatten : mask
      raise ArgumentError, "the action mask allows no action" if mask.first.is_a?(Array) ? mask.any? { |m| m.none? } : flat.none?

      flat
    end
  end
end
