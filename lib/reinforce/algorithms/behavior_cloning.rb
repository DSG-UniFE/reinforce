# frozen_string_literal: true

require 'torch'
require_relative '../networks'

module Reinforce
  module Algorithms
    # ANN-based behavior cloning for discrete action spaces.
    class BehaviorCloning
      attr_reader :policy_model, :logs, :actions

      def initialize(dataset, action_space: nil, learning_rate: 1e-3, hidden_size: 64, policy_model: nil)
        @dataset = dataset
        raise ArgumentError, 'dataset cannot be empty' if @dataset.empty?

        @actions = action_space || @dataset.transitions.map { |t| t[:action] }.uniq
        @action_to_index = {}
        @actions.each_with_index { |action, index| @action_to_index[action] = index }

        @state_size = flatten_state(@dataset.transitions.first[:state]).size
        @policy_model = policy_model || ::Reinforce::Networks.mlp(@state_size, @actions.size, hidden_size: hidden_size)
        @policy_model.train
        @optimizer = Torch::Optim::Adam.new(@policy_model.parameters, lr: learning_rate)
        @criterion = Torch::NN::CrossEntropyLoss.new
        @logs = {loss: []}
      end

      def train(epochs: 20, batch_size: 64)
        epochs.times do
          batch = @dataset.sample([batch_size, @dataset.size].min)
          states = Torch.tensor(batch.map { |transition| flatten_state(transition[:state]) }, dtype: :float32)
          action_indices = Torch.tensor(batch.map { |transition| @action_to_index.fetch(transition[:action]) }, dtype: :int64)

          logits = @policy_model.call(states)
          loss = @criterion.call(logits, action_indices)
          @logs[:loss] << loss.item

          @optimizer.zero_grad
          loss.backward
          @optimizer.step
        end
        self
      end

      def predict(state)
        state_tensor = Torch.tensor([flatten_state(state)], dtype: :float32)
        logits = Torch.no_grad { @policy_model.call(state_tensor) }
        @actions[logits.argmax(1).item]
      end

      def save(path)
        Torch.save(@policy_model.state_dict, path)
      end

      def load(path)
        @policy_model.load_state_dict(Torch.load(path))
        @policy_model.eval
      end

      private

      def flatten_state(state)
        state.is_a?(Array) ? state.flatten.map(&:to_f) : [state.to_f]
      end
    end
  end
end
