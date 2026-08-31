# frozen_string_literal: true

require "torch"
require_relative "../networks"
require_relative "sac"
require_relative "../models/diffusion_policy"
require_relative "../models/diffusion_score_model"
require_relative "../offline/intrinsic_rewards/score_bonus"

module Reinforce
  module Algorithms
    # Exploratory Diffusion Model (ExDM) built on top of the SAC core.
    # ExDM augments rewards with diffusion-score intrinsic bonuses and
    # delegates actor/critic optimization to SAC.
    class ExDM
      include ::Reinforce::Agent

      attr_reader :policy_model, :q_network, :q_target_network, :score_model, :logs, :sac

      def initialize(dataset = nil, action_space: nil, state_size: nil, learning_rate: 3e-4, discount_factor: 0.99,
        entropy_coefficient: 0.1, intrinsic_coefficient: 0.1, hidden_size: 64, noise_std: 0.1,
        policy_model: nil, q1_model: nil, q2_model: nil, q1_target_model: nil, q2_target_model: nil, score_model: nil)
        @dataset = dataset
        if @dataset.nil? && (action_space.nil? || state_size.nil?)
          raise ArgumentError, "action_space and state_size are required when dataset is nil"
        end
        raise ArgumentError, "dataset cannot be empty" if !@dataset.nil? && @dataset.empty?

        @actions = action_space || @dataset.transitions.map { |transition| transition[:action] }.uniq
        @action_to_index = {}
        @actions.each_with_index { |action, index| @action_to_index[action] = index }
        @state_size = state_size || flatten_state(@dataset.transitions.first[:state]).size
        @intrinsic_coefficient = intrinsic_coefficient

        @policy_model = policy_model || ::Reinforce::Models::DiffusionPolicy.new(@state_size, @actions, hidden_size:)
        q1 = q1_model || ::Reinforce::Networks.mlp(@state_size, @actions.size, hidden_size: hidden_size)
        q2 = q2_model || ::Reinforce::Networks.mlp(@state_size, @actions.size, hidden_size: hidden_size)
        q1_target = q1_target_model || ::Reinforce::Networks.mlp(@state_size, @actions.size, hidden_size: hidden_size)
        q2_target = q2_target_model || ::Reinforce::Networks.mlp(@state_size, @actions.size, hidden_size: hidden_size)
        @score_model = score_model || ::Reinforce::Models::DiffusionScoreModel.new(@state_size + @actions.size, hidden_size:)

        @score_optimizer = Torch::Optim::Adam.new(@score_model.parameters, lr: learning_rate)
        @intrinsic_bonus = ::Reinforce::Offline::IntrinsicRewards::ScoreBonus.new(noise_std:)
        @mse = Torch::NN::MSELoss.new

        batch_size = [64, @dataset&.size || 64].min
        @sac = ::Reinforce::Algorithms::SAC.new(
          offline_environment_proxy,
          policy_model: @policy_model,
          q1_model: q1,
          q2_model: q2,
          q1_target_model: q1_target,
          q2_target_model: q2_target,
          learning_rate:,
          discount_factor:,
          entropy_coefficient:,
          auto_entropy_tuning: true,
          batch_size:,
          warmup_steps: 1,
          critic_updates_per_step: 1,
          policy_updates_per_step: 1,
          target_updates_per_step: 1
        )
        @q_network = @sac.q1_model
        @q_target_network = @sac.q1_target_model

        preload_dataset! unless @dataset.nil?
        @logs = {loss: [], policy_loss: [], q_loss: [], score_loss: [], intrinsic_bonus: []}
      end

      def append_transition(state:, action:, reward:, next_state:, done:)
        @sac.append_transition(state:, action:, reward:, next_state:, done:)
      end

      def replay_size
        @sac.replay_size
      end

      def train(epochs: 50, batch_size: 64, tau: 0.05)
        epochs.times do
          train_step_from_replay(batch_size:)
        end
        self
      end

      def train_step_from_replay(batch_size: nil)
        effective_batch_size = batch_size || @sac.batch_size
        return nil unless @sac.ready_for_update?(effective_batch_size)

        batch = @sac.sample_batch(effective_batch_size)
        states, action_indices = tensors_from_batch(batch)
        state_actions = encode_state_actions(states, action_indices)
        intrinsic = @intrinsic_bonus.call(@score_model, state_actions)
        augmented_batch = build_augmented_batch(batch, intrinsic)

        metrics = @sac.train_step(batch: augmented_batch)
        q_loss = (metrics[:q1_loss] + metrics[:q2_loss]) / 2.0
        policy_loss = metrics[:policy_loss]
        score_loss = update_score_model(state_actions)
        total = q_loss + policy_loss + score_loss

        @logs[:loss] << total
        @logs[:q_loss] << q_loss
        @logs[:policy_loss] << policy_loss
        @logs[:score_loss] << score_loss
        @logs[:intrinsic_bonus] << intrinsic.mean.item

        metrics.merge(score_loss:, intrinsic_bonus: intrinsic.mean.item, loss: total)
      end

      def predict(state)
        tensor = Torch.tensor([flatten_state(state)], dtype: :float32)
        @policy_model.predict(tensor)
      end

      private

      def tensors_from_batch(batch)
        states = Torch.tensor(batch.map { |transition| flatten_state(transition[:state]) }, dtype: :float32)
        action_indices = Torch.tensor(batch.map { |transition| action_index(transition) }, dtype: :int64)
        [states, action_indices]
      end

      def encode_state_actions(states, action_indices)
        one_hot = Torch.zeros(action_indices.size(0), @actions.size, dtype: :float32)
        action_indices.to_a.each_with_index do |index, row|
          one_hot[row][index.to_i] = 1.0
        end
        Torch.cat([states, one_hot], dim: 1)
      end

      def update_score_model(state_actions)
        noisy = state_actions + 0.1 * Torch.randn_like(state_actions)
        reconstructed = @score_model.call(noisy)
        loss = @mse.call(reconstructed, state_actions)

        @score_optimizer.zero_grad
        loss.backward
        @score_optimizer.step
        loss.item
      end

      def build_augmented_batch(batch, intrinsic)
        intrinsic_values = intrinsic.to_a
        batch.each_with_index.map do |transition, index|
          base_reward = transition.fetch(:reward, 0.0).to_f
          {
            state: flatten_state(transition[:state]),
            action: transition.key?(:action) ? transition[:action] : @actions[action_index(transition)],
            action_index: action_index(transition),
            reward: base_reward + @intrinsic_coefficient * intrinsic_values[index].to_f,
            next_state: flatten_state(transition[:next_state]),
            done: transition[:done] ? 1.0 : 0.0
          }
        end
      end

      def action_index(transition)
        transition.key?(:action_index) ? transition[:action_index] : @action_to_index.fetch(transition[:action])
      end

      def preload_dataset!
        @dataset.each do |transition|
          append_transition(
            state: transition[:state],
            action: transition[:action],
            reward: transition[:reward],
            next_state: transition[:next_state],
            done: transition[:done]
          )
        end
      end

      def flatten_state(state)
        state.is_a?(Array) ? state.flatten.map(&:to_f) : [state.to_f]
      end

      def offline_environment_proxy
        proxy = Object.new
        actions = @actions
        state_size = @state_size
        proxy.define_singleton_method(:actions) { actions }
        proxy.define_singleton_method(:state_size) { state_size }
        proxy.define_singleton_method(:reset) { raise "offline proxy does not support reset" }
        proxy.define_singleton_method(:step) { |_action| raise "offline proxy does not support step" }
        proxy
      end
    end
  end
end
