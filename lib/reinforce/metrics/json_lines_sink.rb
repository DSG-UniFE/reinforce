# frozen_string_literal: true

require "json"

module Reinforce
  module Metrics
    # Writes every recorded value as one JSON object per line:
    #
    #   {"key":"loss","value":0.0311,"step":41,"time":1762700000.123}
    #
    # A plain, append-only format that is easy to post-process (pandas'
    # `read_json(lines: true)`, `jq`, ...) and to import into experiment
    # trackers such as MLflow or Weights & Biases after the fact.
    #
    # NaN and infinite values are written as the (non-standard, but widely
    # accepted) `NaN`/`Infinity` literals rather than dropped, since a
    # diverging loss is exactly the kind of thing these logs exist to show.
    class JSONLinesSink
      # @parameter destination [String, IO] a file path (truncated and
      #   opened for writing) or an already open IO.
      # @parameter clock [#call] returns the timestamp written with each
      #   value, in seconds since the Unix epoch.
      def initialize(destination, clock: -> { Time.now.to_f })
        @owns_io = !destination.respond_to?(:write)
        @io = @owns_io ? File.open(destination, "w") : destination
        @clock = clock
      end

      def call(key, value, step)
        record = {key: key, value: value, step: step, time: @clock.call}
        @io.puts JSON.generate(record, allow_nan: true)
        @io.flush
      end

      # Close the underlying file, if this sink opened it.
      def close
        @io.close if @owns_io && !@io.closed?
      end
    end
  end
end
