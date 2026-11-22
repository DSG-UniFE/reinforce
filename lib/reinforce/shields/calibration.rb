# frozen_string_literal: true

module Reinforce
  module Shields
    # How well predicted probabilities match outcomes: e.g. a sensor's
    # probability that a cluster accommodates the replicas sent to it,
    # against whether they actually fitted (see docs/sensor-model.md).
    #
    #   calibration = Calibration.new
    #   calibration.add(0.9, true)
    #   calibration.add([0.2, 0.7], [false, true])
    #   calibration.brier # => mean squared error of the probabilities
    #   calibration.expected_calibration_error
    #   calibration.table # => [{low:, high:, predicted:, observed:, count:}, ...]
    class Calibration
      # @parameter bins [Integer] equal-width probability bins for the
      #   expected calibration error and the table.
      def initialize(bins: 10)
        @bins = bins
        @counts = Array.new(bins, 0)
        @predicted = Array.new(bins, 0.0)
        @observed = Array.new(bins, 0)
        @squared_error = 0.0
      end

      # @parameter probabilities [Float, Array(Float)] predicted probabilities.
      # @parameter outcomes [Boolean, Array(Boolean)] whether each event happened.
      def add(probabilities, outcomes)
        Array(probabilities).zip(Array(outcomes)).each do |probability, outcome|
          probability = probability.to_f.clamp(0.0, 1.0)
          hit = outcome ? 1 : 0
          bin = [(probability * @bins).floor, @bins - 1].min
          @counts[bin] += 1
          @predicted[bin] += probability
          @observed[bin] += hit
          @squared_error += (probability - hit)**2
        end
        self
      end

      # @returns [Integer] the predictions added.
      def count = @counts.sum

      # @returns [Float, nil] the Brier score: the mean squared difference
      #   between probabilities and outcomes (0 is perfect).
      def brier
        count.positive? ? @squared_error / count : nil
      end

      # @returns [Float, nil] the mean absolute gap between each bin's mean
      #   prediction and its observed frequency, weighted by the bin's size.
      def expected_calibration_error
        return nil unless count.positive?

        @counts.each_index.sum do |bin|
          next 0.0 if @counts[bin].zero?

          (@counts[bin].to_f / count) * ((@predicted[bin] / @counts[bin]) - (@observed[bin].to_f / @counts[bin])).abs
        end
      end

      # @returns [Array(Hash)] per non-empty bin: its bounds, the mean
      #   prediction, the observed frequency and the count.
      def table
        @counts.each_index.filter_map do |bin|
          next if @counts[bin].zero?

          {low: bin.to_f / @bins, high: (bin + 1).to_f / @bins, predicted: @predicted[bin] / @counts[bin],
           observed: @observed[bin].to_f / @counts[bin], count: @counts[bin]}
        end
      end
    end
  end
end
