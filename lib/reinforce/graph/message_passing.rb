# frozen_string_literal: true

module Reinforce
  module Graph
    # One round of message passing: every edge computes a message from its
    # sender, its receiver and its own features; each receiver combines its
    # messages (sum or mean) and updates its features from them:
    #
    #   m_ij = edge_model([h_i, h_j, e_ij])
    #   h_j' = node_model([h_j, aggregate_i m_ij])
    #
    # Senders and receivers may be different node sets (a bipartite graph,
    # e.g. services sending to locations along placements), of different
    # sizes. With `edge_output_size`, it also returns updated edge
    # features from the messages: e_ij' = edge_head(m_ij).
    #
    # Message passing is permutation-equivariant: reordering nodes (and the
    # edge index with them) reorders the output the same way, and the same
    # weights apply to graphs of any size.
    class MessagePassing < Torch::NN::Module
      attr_reader :output_size

      def initialize(sender_size:, receiver_size: sender_size, edge_size: 0, output_size: receiver_size,
        hidden_size: 64, hidden_layers: 1, reduce: :sum, edge_output_size: nil, activation: Torch::NN::ReLU)
        super()
        raise ArgumentError, "reduce must be :sum or :mean" unless %i[sum mean].include?(reduce)

        @reduce = reduce
        @edge_size = edge_size
        @output_size = output_size
        @edge_model = Reinforce::Networks.mlp(sender_size + receiver_size + edge_size, hidden_size,
          hidden_size:, hidden_layers:, activation:)
        @activation = activation.new
        @node_model = Reinforce::Networks.mlp(receiver_size + hidden_size, output_size, hidden_size:, hidden_layers:, activation:)
        @edge_head = edge_output_size && Torch::NN::Linear.new(hidden_size, edge_output_size)
      end

      # @parameter senders [Torch::Tensor] [senders, sender_size].
      # @parameter receivers [Torch::Tensor] [receivers, receiver_size];
      #   by default the senders (a graph over one node set).
      # @parameter edge_index [Torch::Tensor] [2, edges] long: sender and receiver rows.
      # @parameter edge_features [Torch::Tensor, nil] [edges, edge_size].
      # @returns [Torch::Tensor] the receivers' new features [receivers,
      #   output_size]; with edge_output_size, [those, new edge features].
      def forward(senders, receivers = nil, edge_index:, edge_features: nil)
        receivers ||= senders
        from = edge_index[0]
        to = edge_index[1]
        inputs = [senders.index_select(0, from), receivers.index_select(0, to)]
        if @edge_size.positive?
          raise ArgumentError, "edge_features required (edge_size #{@edge_size})" if edge_features.nil?

          inputs << edge_features
        end
        messages = @activation.call(@edge_model.call(Torch.cat(inputs, 1)))
        aggregated = Graph.aggregate(messages, to, receivers.size(0), reduce: @reduce)
        nodes = @node_model.call(Torch.cat([receivers, aggregated], 1))
        @edge_head ? [nodes, @edge_head.call(messages)] : nodes
      end
    end
  end
end
