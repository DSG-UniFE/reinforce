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

      # Keeps return normalization finite when every return in an episode
      # is identical (zero standard deviation).
      RETURN_NORMALIZATION_EPSILON = 1e-8

      # Default weight of the entropy bonus in #update_policy's loss.
      ENTROPY_COEFFICIENT = 0.01

      attr_reader :metrics

      alias_method :logs, :metrics

      # @parameter environment [Reinforce::Environment] the environment to
      #   train on; #predict returns elements of its `actions`.
      # @parameter model [Torch::NN::Module] the policy network, mapping a
      #   state to one logit per action. Must be a real Torch::NN::Module
      #   (not just something with #forward), since #save/#load go through
      #   its #state_dict.
      # @parameter optimizer [Torch::Optim::Optimizer] built over
      #   `model.parameters`.
      # @parameter normalize_returns [Boolean] standardize each episode's
      #   returns before the update (see #normalize). Skipped for single-step
      #   episodes, whose one return would always normalize to zero and
      #   leave nothing to learn from.
      # @parameter entropy_coefficient [Float] weight of the entropy bonus
      #   (see #update_policy); 0.0 disables it.
      def initialize(environment, model, optimizer, discount_factor: 0.99, normalize_returns: true, entropy_coefficient: ENTROPY_COEFFICIENT)
        @environment = environment
        @actions = environment.actions
        @model = model
        @optimizer = optimizer
        @discount_factor = discount_factor
        @normalize_returns = normalize_returns
        @entropy_coefficient = entropy_coefficient
        @log_probs = []
        @entropies = []
        @rewards = []
        @metrics = Reinforce::Metrics::Recorder.new
      end

      # Train for `episodes` episodes of at most `steps_per_episode` steps
      # each, updating the policy once at the end of every episode (REINFORCE
      # needs the complete episode return before it can update). Same
      # signature as the other online algorithms (see Reinforce::Agent).
      def train(episodes:, steps_per_episode:, **_kwargs)
        episodes.times do
          state = @environment.reset
          episode_reward = 0.0
          episode_length = 0

          steps_per_episode.times do
            action = choose_action(state)
            state, reward, done = @environment.step(action)
            get_reward(reward)
            episode_reward += reward
            episode_length += 1
            break if done
          end

          unless @log_probs.empty?
            @metrics.record(:entropy, Torch.stack(@entropies).mean.item)
            @metrics.record(:loss, update_policy.item)
          end
          @metrics.record(:episode_reward, episode_reward)
          @metrics.record(:episode_length, episode_length)
        end
      end

      def reset
        @log_probs = []
        @entropies = []
        @rewards = []
      end

      # Pick an action from the learned policy, for evaluation. Unlike
      # #choose_action, this records nothing: @log_probs/@rewards are the
      # training buffer for the episode currently being learned from, and
      # evaluating the policy must not leak extra entries into it.
      #
      # @parameter deterministic [Boolean] greedy if true, sampled if false.
      # @returns [Object] an element of the environment's `actions`.
      def predict(state, deterministic: true)
        logits = Torch.no_grad { @model.forward(as_tensor(state)) }
        @actions[CategoricalDistribution.new(logits: logits).choose(deterministic: deterministic)]
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
        @entropies << pd.entropy
        action
      end

      def get_reward(reward)
        @rewards << reward
      end

      # Update the policy from the episode just completed, then clear the
      # episode buffers (REINFORCE is on-policy: these samples can't be
      # reused once the policy that produced them has changed).
      def update_policy
        returns = Torch.tensor(calculate_discounted_rewards(@rewards), dtype: :float32)
        returns = normalize(returns) if @normalize_returns && returns.size(0) > 1
        # Torch.stack, not Torch::Tensor.new: building a fresh tensor from the
        # recorded log-probabilities would copy their values but drop their
        # autograd history, so no gradient could ever reach @model.
        log_probs = Torch.stack(@log_probs)
        # The entropy bonus rewards the policy for staying stochastic. Without
        # it, nothing stops REINFORCE from committing to one action sequence
        # early (e.g. pushing into a wall) and never sampling anything else
        # again -- once the policy is near-deterministic, every episode is the
        # same and the normalized returns carry no signal to escape with.
        loss = Torch.sum(-log_probs * returns) - @entropy_coefficient * Torch.stack(@entropies).sum
        @optimizer.zero_grad
        loss.backward
        @optimizer.step
        reset
        loss
      end

      # Save the policy network's weights, via its state_dict like every
      # other algorithm here. The previous `@model.save(path)` only worked for
      # a model class that happened to define its own #save, which neither
      # Torch::NN::Module nor the example's policy did.
      def save(path)
        Torch.save(@model.state_dict, path)
      end

      # Load weights saved by #save into the (identically shaped) policy
      # network, and switch it to evaluation mode, matching PPO/QFunctionANN.
      def load(path)
        @model.load_state_dict(Torch.load(path))
        @model.eval
      end

      private

      def as_tensor(state)
        state.is_a?(Torch::Tensor) ? state : Torch.tensor(state, dtype: :float32)
      end

      # Standardize an episode's returns to zero mean and unit variance. The
      # mean acts as a simple baseline (actions are reinforced for doing
      # better than this episode's average, not just for any positive
      # return), and dividing by the standard deviation makes the update
      # size independent of the reward scale. Without it, environments whose
      # returns run into the thousands (GridWorld's per-step distance
      # penalty) produce updates large enough to collapse the policy onto
      # one fixed action sequence.
      def normalize(returns)
        (returns - returns.mean) / (returns.std + RETURN_NORMALIZATION_EPSILON)
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
