# frozen_string_literal: true

module Reinforce
  module Shields
    # A shield for discrete (or multi-discrete) actions, wrapping an
    # environment for agents that honor action masks (PPO, DQN): an action is
    # allowed only if a `safety` model gives it a probability of being safe
    # of at least `threshold`.
    #
    # - #action_mask: the environment's mask, minus the options whose safety
    #   probability is below the threshold; per dimension (or for a discrete
    #   space, overall), if none is left, the safest allowed option stays, so
    #   an agent always has a choice.
    # - #step: masks judge options one at a time, so the joint action is
    #   checked again. Below the threshold it is replaced by the first of the
    #   safety model's fallback candidates that reaches it, or, if none does,
    #   by the safest of them (the action itself included).
    #
    # The `safety` model responds to:
    #
    # - `option_probabilities(environment)`: shaped like the environment's
    #   action mask, the probability that each option (holding the others as
    #   they are) is safe;
    # - `probabilities(environment, actions)`: the probability that each
    #   joint action is safe;
    # - `fallbacks(environment, action)`: candidate replacements, safest
    #   preferences first;
    # - `reset(environment)` and `observe(environment, action, result)`
    #   (optional): to follow the episode.
    #
    # #last_info reports the last step's safety probability, whether the
    # shield intervened (the agent's action was below the threshold) and
    # what it executed; #interventions counts them.
    class DiscreteShield
      attr_reader :environment, :safety, :threshold, :last_info, :interventions, :steps

      def initialize(environment, safety, threshold: 0.9)
        @environment = environment
        @safety = safety
        @threshold = threshold
        @interventions = 0
        @steps = 0
        @last_info = {}
      end

      def reset(...)
        state = @environment.reset(...)
        @safety.reset(@environment) if @safety.respond_to?(:reset)
        state
      end

      def state_size = @environment.state_size

      def actions = @environment.actions

      def action_space = @environment.action_space

      def action_mask
        mask = @environment.action_mask
        probabilities = @safety.option_probabilities(@environment)
        if mask.first.is_a?(Array)
          mask.zip(probabilities).map { |m, p| filter(m, p) }
        else
          filter(mask, probabilities)
        end
      end

      def step(action)
        probability = @safety.probabilities(@environment, [action]).first
        executed = action
        executed_probability = probability
        if probability < @threshold
          @interventions += 1
          candidates = @safety.fallbacks(@environment, action)
          scores = @safety.probabilities(@environment, candidates)
          best = scores.each_index.find { |i| scores[i] >= @threshold } || scores.each_index.max_by { |i| scores[i] }
          if best && scores[best] > probability
            executed = candidates[best]
            executed_probability = scores[best]
          end
        end
        @steps += 1
        result = @environment.step(executed)
        @safety.observe(@environment, executed, result) if @safety.respond_to?(:observe)
        @last_info = {probability:, intervened: probability < @threshold, executed:, executed_probability:}
        result
      end

      # Anything else (the wrapped environment's own methods) goes through.
      def method_missing(name, ...) = @environment.respond_to?(name) ? @environment.public_send(name, ...) : super

      def respond_to_missing?(name, include_private = false) = @environment.respond_to?(name, include_private) || super

      private

      def filter(allowed, probabilities)
        safe = allowed.each_index.map { |i| allowed[i] && probabilities[i] >= @threshold }
        return safe if safe.any?

        best = allowed.each_index.select { |i| allowed[i] }.max_by { |i| probabilities[i] }
        allowed.each_index.map { |i| i == best }
      end
    end
  end
end
