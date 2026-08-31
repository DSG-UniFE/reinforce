# frozen_string_literal: true

require "torch"
require_relative "../networks"

module Reinforce
  module Algorithms
    # ANN-based conservative Q-learning (discrete action variant).
    class ConservativeQLearning
      include ::Reinforce::Agent

      attr_reader :q_network, :q_target_network, :actions, :logs

      def initialize(dataset, action_space: nil, learning_rate: 1e-3, discount_factor: 0.99, alpha: 0.1,
        hidden_size: 64, q_network: nil, q_target_network: nil)
        @dataset = dataset
        raise ArgumentError, "dataset cannot be empty" if @dataset.empty?

        @actions = action_space || @dataset.transitions.map { |t| t[:action] }.uniq
        @action_to_index = {}
        @actions.each_with_index { |action, index| @action_to_index[action] = index }

        @state_size = flatten_state(@dataset.transitions.first[:state]).size
        @discount_factor = discount_factor
        @alpha = alpha

        @q_network = q_network || ::Reinforce::Networks.mlp(@state_size, @actions.size, hidden_size: hidden_size)
        @q_target_network = q_target_network || ::Reinforce::Networks.mlp(@state_size, @actions.size, hidden_size: hidden_size)
        @q_target_network.load_state_dict(@q_network.state_dict)
        @q_network.train
        @q_target_network.train

        @optimizer = Torch::Optim::Adam.new(@q_network.parameters, lr: learning_rate)
        @logs = {loss: [], bellman_loss: [], cql_penalty: []}
      end

      def train(epochs: 20, batch_size: 64, tau: 1.0)
        epochs.times do
          batch = @dataset.sample([batch_size, @dataset.size].min)
          states, action_indices, rewards, next_states, dones = tensors_from_batch(batch)

          q_values = @q_network.call(states)
          q_selected = q_values.gather(1, action_indices.reshape(-1, 1)).reshape(-1)

          target_values = nil
          Torch.no_grad do
            next_q_values = @q_target_network.call(next_states)
            max_next = Torch.tensor(next_q_values.to_a.map { |row| row.max }, dtype: :float32)
            target_values = rewards + @discount_factor * (1.0 - dones) * max_next
          end

          bellman_loss = Torch::NN::MSELoss.new.call(q_selected, target_values)
          conservative_penalty = Torch.logsumexp(q_values, dim: 1).mean - q_selected.mean
          loss = bellman_loss + @alpha * conservative_penalty

          @optimizer.zero_grad
          loss.backward
          @optimizer.step

          @logs[:loss] << loss.item
          @logs[:bellman_loss] << bellman_loss.item
          @logs[:cql_penalty] << conservative_penalty.item

          soft_update_targets(tau)
        end
        self
      end

      def predict(state)
        values = q_values_for_state(state)
        @actions[values.each_with_index.max_by { |value, _idx| value }[1]]
      end

      def value(state, action)
        values = q_values_for_state(state)
        values[@action_to_index.fetch(action)]
      end

      def save(path)
        Torch.save(@q_network.state_dict, path)
      end

      def load(path)
        @q_network.load_state_dict(Torch.load(path))
        @q_target_network.load_state_dict(@q_network.state_dict)
        @q_network.eval
        @q_target_network.eval
      end

      private

      def tensors_from_batch(batch)
        states = Torch.tensor(batch.map { |transition| flatten_state(transition[:state]) }, dtype: :float32)
        action_indices = Torch.tensor(batch.map { |transition| @action_to_index.fetch(transition[:action]) }, dtype: :int64)
        rewards = Torch.tensor(batch.map { |transition| transition[:reward].to_f }, dtype: :float32)
        next_states = Torch.tensor(batch.map { |transition| flatten_state(transition[:next_state]) }, dtype: :float32)
        dones = Torch.tensor(batch.map { |transition| transition[:done] ? 1.0 : 0.0 }, dtype: :float32)
        [states, action_indices, rewards, next_states, dones]
      end

      def q_values_for_state(state)
        state_tensor = Torch.tensor([flatten_state(state)], dtype: :float32)
        Torch.no_grad { @q_network.call(state_tensor) }.to_a.first
      end

      def flatten_state(state)
        state.is_a?(Array) ? state.flatten.map(&:to_f) : [state.to_f]
      end

      def soft_update_targets(tau)
        ::Reinforce::Networks.soft_update!(target: @q_target_network, online: @q_network, tau: tau)
      end
    end
  end
end
