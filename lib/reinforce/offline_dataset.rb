# frozen_string_literal: true

module Reinforce
  # Stores offline transitions for dataset-driven RL algorithms.
  class OfflineDataset
    REQUIRED_FIELDS = %i[state action reward next_state done].freeze

    attr_reader :transitions

    def initialize(transitions = [])
      @transitions = []
      transitions.each { |transition| add(transition) }
    end

    def add(transition)
      validate_transition!(transition)
      @transitions << normalize_transition(transition)
    end

    def size
      @transitions.size
    end

    def empty?
      @transitions.empty?
    end

    def sample(batch_size, random: nil)
      raise ArgumentError, 'batch_size must be positive' if batch_size <= 0
      raise ArgumentError, 'cannot sample from an empty dataset' if empty?
      random ||= Random.new

      (0...batch_size).map { @transitions[random.rand(@transitions.size)] }
    end

    def each(&block)
      @transitions.each(&block)
    end

    def initial_states
      tagged = @transitions.select { |transition| transition[:timestep].to_i.zero? }
      states = tagged.map { |transition| transition[:state] }
      states = @transitions.map { |transition| transition[:state] } if states.empty?
      states
    end

    private

    def validate_transition!(transition)
      missing = REQUIRED_FIELDS.reject { |field| transition.key?(field) }
      return if missing.empty?

      raise ArgumentError, "transition is missing required fields: #{missing.join(', ')}"
    end

    def normalize_transition(transition)
      {
        state: deep_copy(transition[:state]),
        action: transition[:action],
        reward: transition[:reward].to_f,
        next_state: deep_copy(transition[:next_state]),
        done: !!transition[:done],
        behavior_prob: transition.fetch(:behavior_prob, nil),
        target_prob: transition.fetch(:target_prob, nil),
        timestep: transition.fetch(:timestep, nil)
      }
    end

    def deep_copy(value)
      value.is_a?(Array) ? value.map { |item| deep_copy(item) } : value
    end
  end
end
