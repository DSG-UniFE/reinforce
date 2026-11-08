# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi.

require "torch"
require_relative "../categorical_distribution"

module Reinforce
  module Algorithms
    # The classic REINFORCE algorithm (Williams, 1992): a Monte Carlo policy
    # gradient method, so named because it updates the policy from complete
    # episode returns rather than bootstrapped estimates. Named
    # MonteCarloPolicyGradient, not REINFORCE/Reinforce, because a bare
    # `Reinforce` class nested under `Reinforce::Algorithms` shadowed the
    # top-level `Reinforce` module for any unqualified `Reinforce::X`
    # reference written elsewhere in this namespace.
    class MonteCarloPolicyGradient
      include ::Reinforce::Agent

      def initialize(num_states, num_actions, discount_factor, model, optimizer)
        @num_states = num_states
        @num_actions = num_actions
        @discount_factor = discount_factor
        @model = model
        @optimizer = optimizer
        @log_probs = []
        @rewards = []
      end

      def reset
        @log_probs = []
        @rewards = []
      end

      # Sample an action from the learned (stochastic) policy, for evaluation.
      # Unlike #choose_action, this records nothing: @log_probs/@rewards are
      # the training buffer for the episode currently being learned from, and
      # evaluating the policy must not leak extra entries into it.
      #
      # @returns [Integer] an action index.
      def predict(state)
        logits = Torch.no_grad { @model.forward(as_tensor(state)) }
        CategoricalDistribution.new(logits: logits).sample.item
      end

      # Sample an action during training, recording its log-probability
      # (still attached to the autograd graph) for #update_policy.
      #
      # @returns [Integer] an action index -- a plain Integer rather than the
      #   0-dim tensor CategoricalDistribution#sample returns, since
      #   environments' #step map Integer indices onto their actions.
      def choose_action(state)
        logits = @model.forward(as_tensor(state))
        pd = CategoricalDistribution.new(logits: logits)
        action = pd.sample.item
        @log_probs << pd.log_probability(action)
        action
      end

      def get_reward(reward)
        @rewards << reward
      end

      # Update the policy from the episode just completed, then clear the
      # episode buffers (REINFORCE is on-policy: these samples can't be
      # reused once the policy that produced them has changed).
      def update_policy
        discounted_rewards = calculate_discounted_rewards(@rewards)
        rewards = Torch::Tensor.new(discounted_rewards)
        # Torch.stack, not Torch::Tensor.new: building a fresh tensor from the
        # recorded log-probabilities would copy their values but drop their
        # autograd history, so no gradient could ever reach @model.
        log_probs = Torch.stack(@log_probs)
        loss = Torch.sum(-log_probs * rewards)
        @optimizer.zero_grad
        loss.backward
        @optimizer.step
        reset
        loss
      end

      # save the model at a given path
      def save(path)
        @model.save(path)
      end

      # load the model at the given path
      def load(path)
        @model.load(path)
      end

      private

      def as_tensor(state)
        state.is_a?(Torch::Tensor) ? state : Torch.tensor(state, dtype: :float32)
      end

      def calculate_discounted_rewards(rewards)
        discounted_rewards = []
        cumulative_reward = 0.0

        rewards.reverse_each do |reward|
          cumulative_reward = reward + @discount_factor * cumulative_reward
          discounted_rewards.unshift(cumulative_reward)
        end

        discounted_rewards
      end
    end
  end
end
