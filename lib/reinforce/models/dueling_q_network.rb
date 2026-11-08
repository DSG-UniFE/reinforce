# frozen_string_literal: true

require "torch"
require_relative "../networks"

module Reinforce
  module Models
    # Dueling network head (Wang et al., 2016): splits Q(s, a) into a
    # state-value stream V(s) and an action-advantage stream A(s, a),
    # sharing a common feature trunk (Reinforce::Networks.trunk), then
    # recombines them as Q(s, a) = V(s) + (A(s, a) - mean_a A(s, a)).
    # Subtracting the mean advantage keeps V and A identifiable (otherwise
    # a constant could shift freely between the two streams and still
    # produce the same Q(s, a), which would make the two streams
    # meaningless on their own).
    #
    # The point of separating them: in states where the choice of action
    # barely matters, V(s) alone can be learned well without having to
    # also disambiguate which action is marginally best, which tends to
    # generalize faster than learning Q(s, a) as one undifferentiated
    # target per action.
    #
    # A drop-in `architecture:` for Reinforce::QFunctionANN -- it exposes
    # the same #forward(state) -> Q-values-per-action interface, so no
    # other code needs to know a network is dueling versus a plain MLP.
    class DuelingQNetwork < Torch::NN::Module
      def initialize(input_size, num_actions, network_config: Reinforce::Networks::Config.new)
        super()
        layers, trunk_output_size = Reinforce::Networks.build_hidden_layers(
          input_size, network_config.hidden_size, network_config.hidden_layers, network_config.activation
        )
        @shared = Torch::NN::Sequential.new(*layers)
        @value_head = Torch::NN::Linear.new(trunk_output_size, 1)
        @advantage_head = Torch::NN::Linear.new(trunk_output_size, num_actions)
        to(Reinforce::Device.resolve(network_config.device))
      end

      def forward(state)
        features = @shared.call(state)
        value = @value_head.call(features)
        advantage = @advantage_head.call(features)
        # dim: -1, not a hardcoded dim: 1 -- callers pass both unbatched
        # states ([num_actions], e.g. DQN#choose_action acting on a single
        # state) and batched ones ([batch, num_actions], e.g. #update
        # training on a minibatch). -1 is the action dimension either way;
        # 1 would raise on the unbatched shape, which has no dimension 1.
        value + (advantage - advantage.mean(dim: -1, keepdim: true))
      end
    end
  end
end
