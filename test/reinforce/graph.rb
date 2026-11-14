# frozen_string_literal: true

require "reinforce/graph"

describe Reinforce::Graph do
  let(:edge_index) { Torch.tensor([[0, 1, 2, 2], [1, 2, 0, 1]], dtype: :long) }

  it "aggregates messages per receiver, by sum or mean" do
    messages = Torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    expect(Reinforce::Graph.aggregate(messages, edge_index[1], 4).to_a).to be == [[3.0], [5.0], [2.0], [0.0]]
    expect(Reinforce::Graph.aggregate(messages, edge_index[1], 4, reduce: :mean).to_a).to be == [[3.0], [2.5], [2.0], [0.0]]
  end

  it "adds reverse edges" do
    both, features = Reinforce::Graph.bidirectional(Torch.tensor([[0], [1]], dtype: :long), Torch.tensor([[7.0]]))
    expect(both.to_a).to be == [[0, 1], [1, 0]]
    expect(features.to_a).to be == [[7.0], [7.0]]
  end

  it "batches graphs with offset edges, bipartite ones included" do
    a = {senders: Torch.zeros(2, 1), receivers: Torch.zeros(3, 1), edge_index: Torch.tensor([[0, 1], [2, 0]], dtype: :long)}
    b = {senders: Torch.zeros(1, 1), receivers: Torch.zeros(2, 1), edge_index: Torch.tensor([[0], [1]], dtype: :long)}
    batch = Reinforce::Graph.batch([a, b])

    expect(batch[:edge_index].to_a).to be == [[0, 1, 2], [2, 0, 4]]
    expect(batch[:senders].size(0)).to be == 3
    expect(batch[:receivers].size(0)).to be == 5
    expect(batch[:graph].to_a).to be == [0, 0, 0, 1, 1]
  end
end

describe Reinforce::Graph::MessagePassing do
  let(:edge_index) { Torch.tensor([[0, 1, 2, 2], [1, 2, 0, 1]], dtype: :long) }

  it "updates the receivers from their messages, on one node set or two" do
    Torch.manual_seed(1)
    layer = Reinforce::Graph::MessagePassing.new(sender_size: 3, edge_size: 2, output_size: 5)
    expect(layer.call(Torch.randn(3, 3), edge_index:, edge_features: Torch.randn(4, 2)).shape).to be == [3, 5]

    bipartite = Reinforce::Graph::MessagePassing.new(sender_size: 3, receiver_size: 4, output_size: 2, edge_output_size: 6)
    nodes, edges = bipartite.call(Torch.randn(3, 3), Torch.randn(5, 4), edge_index: Torch.tensor([[0, 2, 2], [4, 0, 4]], dtype: :long))
    expect([nodes.shape, edges.shape]).to be == [[5, 2], [3, 6]]
  end

  it "is permutation-equivariant" do
    Torch.manual_seed(2)
    layer = Reinforce::Graph::MessagePassing.new(sender_size: 3, output_size: 4)
    nodes = Torch.randn(3, 3)
    output = layer.call(nodes, edge_index:)
    order = [2, 0, 1] # new row k holds old node order[k]
    position = order.each_with_index.to_h # old node -> new row
    permuted_edges = Torch.tensor(edge_index.to_a.map { |row| row.map { |n| position[n] } }, dtype: :long)
    permuted = layer.call(nodes.index_select(0, Torch.tensor(order)), edge_index: permuted_edges)

    order.each_with_index do |old, new_row|
      permuted[new_row].to_a.zip(output[old].to_a).each { |a, b| expect(a).to be_within(1e-5).of(b) }
    end
  end

  it "passes gradients to every parameter" do
    layer = Reinforce::Graph::MessagePassing.new(sender_size: 2, edge_size: 1, output_size: 1, edge_output_size: 1)
    nodes, edges = layer.call(Torch.randn(3, 2), edge_index:, edge_features: Torch.randn(4, 1))
    (nodes.sum + edges.sum).backward
    expect(layer.parameters.all? { |p| p.grad && p.grad.abs.sum.item.positive? }).to be == true
  end

  it "learns to sum its neighbours' values" do
    Torch.manual_seed(3)
    layer = Reinforce::Graph::MessagePassing.new(sender_size: 1, output_size: 1, hidden_size: 16)
    optimizer = Torch::Optim::Adam.new(layer.parameters, lr: 0.01)
    targets = ->(x) { Reinforce::Graph.aggregate(x.index_select(0, edge_index[0]), edge_index[1], 3) }
    loss = nil
    400.times do
      x = Torch.randn(3, 1)
      loss = (layer.call(x, edge_index:) - targets.call(x)).pow(2).mean
      optimizer.zero_grad
      loss.backward
      optimizer.step
    end
    expect(loss.item).to be < 0.05
  end
end

describe Reinforce::Graph::Temporal do
  it "keeps a state per node across steps" do
    Torch.manual_seed(4)
    temporal = Reinforce::Graph::Temporal.new(2, 8)
    states = temporal.unroll(Torch.randn(5, 3, 2))
    expect(states.shape).to be == [5, 3, 8]
    expect(temporal.call(Torch.randn(3, 2), states[-1]).shape).to be == [3, 8]
  end

  it "learns to remember the first input of a sequence" do
    Torch.manual_seed(5)
    temporal = Reinforce::Graph::Temporal.new(1, 16)
    readout = Torch::NN::Linear.new(16, 1)
    optimizer = Torch::Optim::Adam.new(temporal.parameters + readout.parameters, lr: 0.01)
    loss = nil
    500.times do
      first = Torch.randn(4, 1)
      sequence = Torch.cat([first.unsqueeze(0), Torch.zeros(5, 4, 1)], 0)
      loss = (readout.call(temporal.unroll(sequence)[-1]) - first).pow(2).mean
      optimizer.zero_grad
      loss.backward
      optimizer.step
    end
    expect(loss.item).to be < 0.05
  end
end
