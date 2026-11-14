# frozen_string_literal: true

module Reinforce
  module Graph
    # A gated recurrent unit cell (Cho et al., 2014), as torch-rb lacks
    # one: from an input [batch, input_size] and the previous state [batch,
    # hidden_size], the next state
    #
    #   r = sigmoid(W_r x + U_r h), z = sigmoid(W_z x + U_z h)
    #   n = tanh(W_n x + r * (U_n h)), h' = (1 - z) * n + z * h
    class GRUCell < Torch::NN::Module
      attr_reader :hidden_size

      def initialize(input_size, hidden_size)
        super()
        @hidden_size = hidden_size
        @input_gates = Torch::NN::Linear.new(input_size, 3 * hidden_size)
        @hidden_gates = Torch::NN::Linear.new(hidden_size, 3 * hidden_size)
      end

      def forward(input, state = nil)
        state ||= Torch.zeros(input.size(0), @hidden_size, dtype: input.dtype, device: input.device)
        x_r, x_z, x_n = @input_gates.call(input).chunk(3, 1)
        h_r, h_z, h_n = @hidden_gates.call(state).chunk(3, 1)
        reset = Torch.sigmoid(x_r + h_r)
        update = Torch.sigmoid(x_z + h_z)
        candidate = Torch.tanh(x_n + reset * h_n)
        (1 - update) * candidate + update * state
      end
    end

    # A recurrent state per node: each step, every node's state is updated
    # from its new features (for example the output of message passing)
    # with a shared GRU cell, so the same weights follow nodes in graphs of
    # any size. Nodes must keep their rows from one step to the next.
    class Temporal < Torch::NN::Module
      def initialize(input_size, hidden_size)
        super()
        @cell = GRUCell.new(input_size, hidden_size)
      end

      def hidden_size = @cell.hidden_size

      # @parameter inputs [Torch::Tensor] [nodes, input_size] this step's features.
      # @parameter state [Torch::Tensor, nil] [nodes, hidden_size], nil at the start.
      # @returns [Torch::Tensor] the new state [nodes, hidden_size].
      def forward(inputs, state = nil) = @cell.call(inputs, state)

      # Runs a sequence [steps, nodes, input_size] from `state`, returning
      # every step's state [steps, nodes, hidden_size].
      def unroll(sequence, state = nil)
        states = sequence.unbind(0).map { |inputs| state = forward(inputs, state) }
        Torch.stack(states, 0)
      end
    end
  end
end
