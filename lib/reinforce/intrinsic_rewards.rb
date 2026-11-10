# frozen_string_literal: true

require_relative "intrinsic_rewards/count_based"

module Reinforce
  # Exploration bonuses ("intrinsic rewards") that online algorithms can
  # add to the environment's reward. Pass one as `intrinsic_reward:` to
  # PPO, DQN or SAC. Every bonus implements:
  #
  # - #observe(state): called with each state the agent reaches while
  #   training, to update the bonus's statistics (e.g. visit counts);
  # - #call(states): the current bonus for an Array of states, as an Array
  #   of Floats, already scaled by the bonus's own coefficient;
  # - #train(states): fits a learned bonus (e.g. RND's predictor) on a batch
  #   of states; a no-op for bonuses with nothing to learn.
  #
  # The bonus is for the state a transition leads to (its next state).
  # On-policy algorithms (PPO) add it as they collect each rollout;
  # off-policy algorithms (DQN, SAC) store raw rewards and add the bonus
  # when they sample a batch, so replayed transitions get the bonus under
  # the *current* statistics rather than the ones from when they were
  # stored. Metrics' `episode_reward` stays the extrinsic reward; the bonus
  # is recorded separately as `intrinsic_reward`.
  module IntrinsicRewards
  end
end
