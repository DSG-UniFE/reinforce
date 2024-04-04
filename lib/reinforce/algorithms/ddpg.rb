## frozen_string_literal: true

require 'torch'

module Reinforce
  module Algorithms
    class QNetwork < Torch::NN::Module
      def initialize(state_size, num_actions)
        super()
        @fc1 = Torch::NN::Linear.new(state_size + num_actions, 64)
        @fc2 = Torch::NN::Linear.new(64, 64)
        @fc3 = Torch::NN::Linear.new(64, num_actions)
      end
      def forward(x, a)
        x = Torch.cat([x, a], dim: 1)
        x = Torch::NN::Functional.relu(@fc1.call(x))
        x = Torch::NN::Functional.relu(@fc2.call(x))
        x = @fc3.call(x)
      end
    end

    class Actor < Torch::NN::Module
      def initialize(state_size, num_actions)
        super()
        @fc1 = Torch::NN::Linear.new(state_size, 64)
        @fc2 = Torch::NN::Linear.new(64, 64)
        @fc3 = Torch::NN::Linear.new(64, num_actions)
      end

      def forward(x)
        x = Torch::NN::Functional.relu(@fc1.call(x))
        x = Torch::NN::Functional.relu(@fc2.call(x))
        x = @fc3.call(x)
      end
    end

    class DDPG 
      def initialize(state_size, num_actions, actor_lr: 0.0001, critic_lr: 0.001, gamma: 0.99, tau: 0.001)
        @actor = Actor.new(state_size, num_actions)
        @actor_target = Actor.new(state_size, num_actions)
        @critic = QNetwork.new(state_size, num_actions)
        @critic_target = QNetwork.new(state_size, num_actions)
        @actor_optimizer = Torch::Optim::Adam.new(@actor.parameters, lr: actor_lr)
        @critic_optimizer = Torch::Optim::Adam.new(@critic.parameters, lr: critic_lr)
        @gamma = gamma
        @tau = tau
        @logs = {}
      end

      def train(n_episodes, max_actions_per_episode)
        n_episodes.times do |episode|
          state = @environment.reset
          episode_reward = 0
          max_actions_per_episode.times do |action|
        action = choose_action(state)
        next_state, reward, done = @environment.step(action)
        episode_reward += reward
        experience = Experience.new(state, action, reward, next_state, done)
        store_experience(experience)
        state = next_state
        update()
        break if done
          end
          @logs[:episode_reward] << episode_reward
          @logs[:episode_length] << action
        end
      end

      def update
        batch = @experience_replay.sample
        states, actions, rewards, next_states, dones = batch
        update_critic(states, actions, rewards, next_states, dones)
        update_actor(states)
        update_target_networks()
      end

      def update_actor(states)
        @actor_optimizer.zero_grad
        loss = -@critic.call(states, @actor.call(states)).mean
        loss.backward
        @actor_optimizer.step
      end

      def update_critic(states, actions, rewards, next_states, dones)
        @critic_optimizer.zero_grad
        target = rewards + @gamma * @critic_target.call(next_states, @actor_target.call(next_states)) * (1 - dones)
        loss = Torch::NN::Functional.mse_loss(@critic.call(states, actions), target)
        loss.backward
        @critic_optimizer.step
      end

      def update_target_networks
        soft_update(@actor_target, @actor)
        soft_update(@critic_target, @critic)
      end

      def soft_update(target, source)
        target.parameters.each_with_index do |target_param, i|
          source_param = source.parameters[i]
          target_param.data = target_param.data * (1 - @tau) + source_param.data * @tau
        end
      end

    end

  end
  
end
