# frozen_string_literal: true

require "torch"
require_relative "simplex"

module Reinforce
  module Shields
    # A shield for allocation actions, whose components in [-1, 1] stand for
    # shares of a resource (see Reinforce::Environments::MultiCluster), for
    # Reinforce::Algorithms::ContinuousSAC's `shield:`.
    #
    # An action a becomes shares q0 by Euclidean projection of (a + 1) / 2
    # onto the simplex, which a `safety` model scores in [0, 1]. Shares
    # scoring at least `threshold` are kept. Otherwise the shield runs
    # `search_steps` steps of gradient ascent from q0 (on unconstrained
    # values, projected onto the simplex), maximizing
    #
    #   score(q) - distance_weight * |q - q0|^2   while score(q) < threshold,
    #   -cost(q)                                    once score(q) >= threshold,
    #
    # where cost(q) = |q - q0|^2 + preference_weight * preference(q). Of q0
    # and the shares after each step, it executes the cheapest that reaches
    # the threshold, or, if none does, the safest (the cheapest among equally
    # safe ones), as 2q - 1. Projected shares are executed even when q0 is
    # kept, as they may differ from the environment's own normalization.
    #
    # For training, #safety_loss is -log(max(score, probability_floor)) of
    # the actor's deterministic actions, averaged over a batch: differentiable
    # through the safety model into the actor.
    #
    # The `safety` model responds to:
    #
    # - `score(observations, shares)`: batch x observation size and batch x
    #   components tensors to a batch tensor in [0, 1], differentiable in
    #   the shares (e.g. MultiClusterSafety);
    # - `preference(observations, shares)` (optional): a batch tensor of
    #   costs for the shares, other than departing from q0 (e.g.
    #   MultiClusterSafety#latency); 0 without it.
    class AllocationShield
      # Safety scores within this tolerance count as equal when choosing
      # among unsafe candidates.
      SCORE_TOLERANCE = 1e-9

      attr_reader :safety, :threshold, :search_steps, :step_size, :distance_weight, :preference_weight, :probability_floor

      # @returns [Hash{Symbol => Numeric}] about the last #call: whether the
      #   base shares were below the threshold (`interventions`), whether
      #   other shares were executed (`corrections`), and the score of the
      #   executed shares (`safety`).
      attr_reader :last_info

      # @parameter safety [#score] the safety model.
      # @parameter preference [Symbol, nil] the name of the safety model's
      #   method giving the preference cost, if any.
      def initialize(safety, preference: nil, threshold: 0.6, search_steps: 6, step_size: 0.5, distance_weight: 0.05,
        preference_weight: 0.5, probability_floor: 1e-8)
        @safety = safety
        @preference = preference
        @threshold = threshold
        @search_steps = search_steps
        @step_size = step_size
        @distance_weight = distance_weight
        @preference_weight = preference_weight
        @probability_floor = probability_floor
        @last_info = {}
      end

      # @parameter state [Array(Numeric)] an observation.
      # @parameter action [Array(Numeric)] the base action, in [-1, 1].
      # @returns [Array(Float)] the action to execute, in [-1, 1].
      def call(state, action)
        observation = Torch.tensor([state.flatten.map(&:to_f)], dtype: :float32)
        base = Torch.no_grad { Simplex.project((Torch.tensor([action.map(&:to_f)], dtype: :float32) + 1.0) / 2.0) }
        base_score = Torch.no_grad { @safety.score(observation, base) }.item

        chosen, chosen_score = if base_score >= @threshold
          [base, base_score]
        else
          choose(search(observation, base, base_score))
        end
        @last_info = {interventions: (base_score < @threshold) ? 1 : 0, corrections: chosen.equal?(base) ? 0 : 1, safety: chosen_score}
        ((chosen * 2.0) - 1.0)[0].to_a
      end

      # @parameter states [Torch::Tensor] batch x observation size.
      # @parameter actions [Torch::Tensor] batch x components, in [-1, 1].
      # @returns [Torch::Tensor] the mean safety loss (a scalar).
      def safety_loss(states, actions)
        scores = @safety.score(states, Simplex.project((actions + 1.0) / 2.0))
        (-Torch.log(Torch.clamp(scores, min: @probability_floor))).mean
      end

      private

      # @returns [Array(Array(Torch::Tensor, Float, Float))] the candidates,
      #   as [shares, score, cost], q0 first.
      def search(observation, base, base_score)
        candidates = [[base, base_score, cost(observation, base, base).item]]
        values = base.clone
        @search_steps.times do
          values.requires_grad!(true)
          shares = Simplex.project(values)
          score = @safety.score(observation, shares)
          objective = if score.item < @threshold
            score - (((shares - base)**2).sum(1) * @distance_weight)
          else
            -cost(observation, shares, base)
          end
          objective.sum.backward
          values = (values + (values.grad * @step_size)).detach

          Torch.no_grad do
            shares = Simplex.project(values)
            candidates << [shares, @safety.score(observation, shares).item, cost(observation, shares, base).item]
          end
        end
        candidates
      end

      def choose(candidates)
        safe = candidates.select { |_shares, score, _cost| score >= @threshold }
        best = if safe.any?
          safe.min_by { |_shares, _score, cost| cost }
        else
          top = candidates.map { |_shares, score, _cost| score }.max
          candidates.select { |_shares, score, _cost| score >= top - SCORE_TOLERANCE }.min_by { |_shares, _score, cost| cost }
        end
        best.first(2)
      end

      def cost(observation, shares, base)
        distance = ((shares - base)**2).sum(1)
        return distance unless @preference

        distance + (@safety.public_send(@preference, observation, shares) * @preference_weight)
      end
    end
  end
end
