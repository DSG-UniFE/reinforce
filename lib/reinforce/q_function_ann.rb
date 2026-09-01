# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi, Filippo Poltronieri

require "torch"
require "forwardable"
require_relative "categorical_distribution"
require_relative "networks"

module Reinforce
  # input to the network is the current state
  # output of the network is the log probabilities of each action
  class QFunctionANN
    extend Forwardable

    def_delegators :@architecture, :apply, :parameters, :state_dict, :load_state_dict
    attr_reader :optimizer, :architecture

    def initialize(state_size, num_actions, learning_rate, discount_factor, architecture: nil)
      @num_actions = num_actions
      # 512-wide, 2 hidden layers: this default predates Reinforce::Networks
      # and is kept as-is so existing callers/examples that rely on it are
      # unaffected; everywhere else in the library defaults to hidden_size: 64.
      @architecture = architecture || Reinforce::Networks.mlp(state_size, num_actions, hidden_size: 512)
      @architecture.train # Enable training mode
      # Create the optimizer
      @optimizer = Torch::Optim::Adam.new(@architecture.parameters, lr: learning_rate)
      @discount_factor = discount_factor
    end

    def forward(state)
      argument = if state.is_a?(Torch::Tensor)
        state
      else
        Torch::Tensor.new(state)
      end
      @architecture.forward(argument)
    end

    def get_action(state)
      argument = if state.is_a?(Torch::Tensor)
        state
      else
        Torch::Tensor.new(state)
      end
      logits = Torch.no_grad { forward(argument) }
      CategoricalDistribution.new(logits: logits).sample
    end

    def random_action(_state)
      rand(@num_actions)
    end

    # Perform one TD update from a batch of experience.
    #
    # @parameter experience [Hash] a batch with :state, :action, :next_state,
    #   :reward, :done, and (if on_policy: true) :next_action keys -- the
    #   shape Reinforce::Experience#sample and
    #   Reinforce::PrioritizedExperienceReplay#sample both return.
    # @parameter on_policy [Boolean] bootstrap from experience[:next_action]
    #   (a genuine SARSA target) instead of a greedy or Double-DQN-selected
    #   action. Mirrors Reinforce::Algorithms::TemporalDifference#learn's
    #   on_policy: flag for the tabular case. Takes priority over
    #   double_dqn: if both are given -- an on-policy algorithm's next
    #   action comes from the behavior policy, not from any Q-network.
    # @parameter target [#forward, nil] an optional separate network to
    #   bootstrap next-state Q-values from (e.g. DQN's slowly-updated
    #   target network -- see Reinforce::Networks.soft_update!). Defaults
    #   to bootstrapping from this network itself, which is what a
    #   target-network-free caller like SARSA wants.
    # @parameter double_dqn [Boolean] when true (and target: is given),
    #   selects the bootstrap action via this (online) network's argmax but
    #   evaluates its value using target's Q-values instead of this
    #   network's own -- Van Hasselt et al. (2015)'s fix for vanilla DQN's
    #   overestimation bias, which comes from the same noisy estimator
    #   both picking the "best" action and judging how good it is.
    # @parameter weights [Array, Torch::Tensor, nil] optional per-sample
    #   importance-sampling weights (e.g. from
    #   Reinforce::PrioritizedExperienceReplay#sample), applied to the loss
    #   so priority-proportional sampling doesn't bias the expected update.
    #   Defaults to unweighted (uniform) loss.
    # @returns [Hash] `{loss:, td_errors:}` -- td_errors are the raw
    #   (signed) target-minus-prediction values, e.g. for
    #   Reinforce::PrioritizedExperienceReplay#update_priorities.
    def update(experience, on_policy: false, target: nil, double_dqn: false, weights: nil)
      # Need to tell Torch not to track the gradient for these operations.
      # See L. Graesser, W.L. Keng, "Foundations of Deep Reinforcement
      # Learning", Section 3.5.2, page 70.
      bootstrap_source = target || self
      next_q_values = Torch.no_grad { bootstrap_source.forward(experience[:next_state]) }

      # `next_action` entries may be plain Ruby integers or 0-dim Torch
      # tensors depending on which policy produced them, hence #to_i.
      next_actions = if on_policy
        experience[:next_action].map(&:to_i)
      elsif double_dqn
        raise ArgumentError, "double_dqn: true requires target:" unless target

        online_next_q_values = Torch.no_grad { forward(experience[:next_state]) }
        online_next_q_values.argmax(1).to_a
      else
        next_q_values.argmax(1).to_a
      end

      target_values = compute_td_targets(next_q_values, next_actions, experience[:reward], experience[:done])

      predicted_q_values = forward(experience[:state])
      taken_q_values = q_values_for_actions(predicted_q_values, experience[:action])

      criterion = Torch::NN::MSELoss.new(reduction: "none")
      per_sample_loss = criterion.call(taken_q_values, target_values)
      per_sample_loss *= Torch.tensor(weights, dtype: :float32) if weights
      loss = per_sample_loss.mean
      td_errors = (target_values - taken_q_values).detach.to_a

      @optimizer.zero_grad
      loss.backward
      @optimizer.step

      {loss: loss.item, td_errors: td_errors}
    end

    # Compute TD targets `reward + discount_factor * Q(next_state,
    # next_action)`, or just `reward` on terminal transitions.
    # `next_actions` picks which column of `next_q_values` to bootstrap
    # from for each sample -- see #update for how that selection differs
    # between on-policy, Double DQN, and vanilla (Q-learning) targets.
    def compute_td_targets(next_q_values, next_actions, rewards, dones)
      targets = Torch.zeros(rewards.size)
      next_actions.zip(rewards, dones).each_with_index do |(next_action, reward, done), i|
        targets[i] = done ? reward : reward + @discount_factor * next_q_values[i][next_action]
      end
      targets
    end

    # Gather Q(s, a) for a batch of taken actions from a batch of
    # per-action Q-value rows.
    def q_values_for_actions(q_values, actions)
      indices = actions.is_a?(Torch::Tensor) ? actions : Torch.tensor(actions)
      q_values.gather(1, indices.long.reshape(-1, 1)).reshape(-1)
    end

    def soft_update(q_network, tau)
      # q_network may be another QFunctionANN (which delegates #state_dict
      # to its own @architecture, see def_delegators above) or a raw
      # Torch::NN::Module -- Networks.soft_update! only needs #state_dict
      # (and, on the target side, #load_state_dict), so either works.
      Reinforce::Networks.soft_update!(target: @architecture, online: q_network, tau: tau)
    end

    def save(path)
      Torch.save(@architecture.state_dict, path)
    end

    def load(path)
      @architecture.load_state_dict(Torch.load(path))
      @architecture.eval
    end
  end
end
