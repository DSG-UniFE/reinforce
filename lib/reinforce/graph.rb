# frozen_string_literal: true

require "torch"
require_relative "networks"
require_relative "graph/message_passing"
require_relative "graph/temporal"

module Reinforce
  # Building blocks for graph neural networks (Battaglia et al., 2018,
  # "Relational inductive biases, deep learning, and graph networks"):
  # message passing over the edges of a graph, possibly between two node
  # sets (a bipartite graph), and a recurrent state per node over time.
  #
  # A graph is given as tensors: node features [nodes, features], an edge
  # index [2, edges] of Integer (long) sender and receiver rows, and
  # optionally edge features [edges, features]. Many graphs train fastest
  # as one: see .batch.
  module Graph
    module_function

    # Combines the messages arriving at each of `size` receivers.
    #
    # @parameter messages [Torch::Tensor] [edges, features].
    # @parameter receivers [Torch::Tensor] [edges] the receiver of each message (long).
    # @parameter reduce [Symbol] :sum or :mean (receivers without messages get zeros).
    # @returns [Torch::Tensor] [size, features].
    def aggregate(messages, receivers, size, reduce: :sum)
      total = Torch.zeros(size, messages.size(1), dtype: messages.dtype, device: messages.device)
        .index_add(0, receivers, messages)
      return total if reduce == :sum
      raise ArgumentError, "reduce must be :sum or :mean" unless reduce == :mean

      counts = Torch.zeros(size, dtype: messages.dtype, device: messages.device)
        .index_add(0, receivers, Torch.ones(receivers.size(0), dtype: messages.dtype, device: messages.device))
      total / counts.clamp(min: 1.0).unsqueeze(1)
    end

    # An undirected graph's edges in both directions: the edge index with
    # its reversed copy appended, and the edge features repeated.
    def bidirectional(edge_index, edge_features = nil)
      both = Torch.cat([edge_index, edge_index.flip(0)], 1)
      [both, edge_features && Torch.cat([edge_features, edge_features], 0)]
    end

    # Several graphs as one disconnected graph: node features concatenated,
    # edge indexes offset by the senders' and receivers' positions.
    #
    # @parameter graphs [Array(Hash)] each {senders:, receivers: (default:
    #   senders), edge_index:, edge_features: (optional)}, where senders and
    #   receivers are node feature tensors.
    # @returns [Hash] the same keys for the batch, plus :graph (each
    #   receiver's graph, long), for pooling per graph.
    def batch(graphs)
      sender_offset = 0
      receiver_offset = 0
      indexes = []
      owners = []
      graphs.each_with_index do |graph, i|
        senders = graph.fetch(:senders)
        receivers = graph.fetch(:receivers, senders)
        offset = Torch.tensor([[sender_offset], [receiver_offset]], dtype: :long)
        indexes << graph.fetch(:edge_index) + offset
        owners << Torch.full([receivers.size(0)], i, dtype: :long)
        sender_offset += senders.size(0)
        receiver_offset += receivers.size(0)
      end
      bipartite = graphs.any? { |g| g.key?(:receivers) }
      {
        senders: Torch.cat(graphs.map { |g| g[:senders] }, 0),
        receivers: bipartite ? Torch.cat(graphs.map { |g| g.fetch(:receivers, g[:senders]) }, 0) : nil,
        edge_index: Torch.cat(indexes, 1),
        edge_features: graphs.first[:edge_features] && Torch.cat(graphs.map { |g| g[:edge_features] }, 0),
        graph: Torch.cat(owners, 0)
      }.compact
    end
  end
end
