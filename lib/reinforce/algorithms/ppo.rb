# frozen_string_literal: true

require_relative "../experience"
require_relative "../categorical_distribution"
require_relative "../networks"
require "torch"

module Reinforce
  module Algorithms
    # PPO's actor-critic network: a policy ("actor") model and a value
    # ("critic") model sharing one training step. Named
    # ActorCriticNetwork, not Agent, because a bare `Agent` class nested
    # under `Reinforce::Algorithms` shadowed the top-level
    # `Reinforce::Agent` contract module for any unqualified
    # `Reinforce::Agent` reference written elsewhere in this namespace.
    class ActorCriticNetwork < Torch::NN::Module
      # cleanrl-style PPO defaults: Tanh (not Reinforce::Networks::Config's
      # usual ReLU) is the standard activation for PPO's actor/critic MLPs.
      DEFAULT_NETWORK_CONFIG = Reinforce::Networks::Config.new(activation: Torch::NN::Tanh)

      attr_reader :policy_model, :value_model

      def initialize(state_size, num_actions, policy_model = nil, value_model = nil,
        network_config: DEFAULT_NETWORK_CONFIG)
        super()
        if policy_model.nil? || value_model.nil?
          @policy_model = build_head(state_size, num_actions, network_config, output_std: 0.01)
          @value_model = build_head(state_size, 1, network_config, output_std: 1.0)
        else
          @policy_model = policy_model
          @value_model = value_model
        end
        [@policy_model, @value_model].each(&:train)
        to(Reinforce::Device.resolve(network_config.device))
      end

      # from cleanrl
      def layer_init(layer, std = Math.sqrt(2), bias_const = 0.0)
        Torch::NN::Init.orthogonal!(layer.weight, gain: std)
        Torch::NN::Init.constant!(layer.bias, bias_const)
        layer
      end

      def get_value(x)
        @value_model.call(x)
      end

      def get_action_and_value(x, action = nil)
        logits = @policy_model.call(x)
        # warn "logits: #{logits}"
        pd = CategoricalDistribution.new(logits: logits)
        action = pd.sample if action.nil?
        value = @value_model.call(x)
        logprob = pd.log_probability(action)
        entropy = pd.entropy
        [action, logprob, entropy, value]
      end

      private

      # Builds one Reinforce::Networks-shaped MLP head with cleanrl's
      # per-layer orthogonal initialization: std = sqrt(2) (the
      # near-standard gain for Tanh/ReLU hidden layers) on every hidden
      # layer, and a much smaller `output_std` on the final layer --
      # 0.01 for the policy head keeps the initial action distribution
      # close to uniform, 1.0 for the value head keeps early value
      # estimates from being either saturated or absurdly large. Losing
      # this scheme would genuinely hurt PPO's early training stability,
      # so it's preserved exactly even though the hidden layers themselves
      # now come from the shared Reinforce::Networks primitive.
      def build_head(input_size, output_size, network_config, output_std:)
        layers, trunk_output_size = Reinforce::Networks.build_hidden_layers(
          input_size, network_config.hidden_size, network_config.hidden_layers, network_config.activation
        )
        layers.each { |layer| layer_init(layer) if layer.is_a?(Torch::NN::Linear) }
        layers << layer_init(Torch::NN::Linear.new(trunk_output_size, output_size), output_std)
        Torch::NN::Sequential.new(*layers)
      end
    end

    class PPO
      include ::Reinforce::Agent

      attr_reader :logs
      attr_accessor :agent, :optimizer

      def initialize(environment, learning_rate, policy = nil, value = nil, clip_param = 0.2, ppo_epochs = 10, minibatch_size = 32, discount_factor = 0.99,
        network_config: ActorCriticNetwork::DEFAULT_NETWORK_CONFIG, entropy_coefficient: 0.01)
        @environment = environment
        @entropy_coefficient = entropy_coefficient
        @agent = ActorCriticNetwork.new(
          environment.state_size, environment.actions.size, policy, value, network_config: network_config
        )
        @gaelam = 0.97
        @clip_param = clip_param
        @ppo_epochs = ppo_epochs
        @minibatch_size = minibatch_size
        @discount_factor = discount_factor
        @learning_rate = learning_rate
        # Create the optimizer
        @logs = {loss: [], episode_reward: [], episode_length: []}
        @optimizer = Torch::Optim::Adam.new([@agent.policy_model.parameters, @agent.value_model.parameters].flatten, lr: learning_rate, eps: 1e-5)
      end

      def save(filename)
        Torch.save(@agent.state_dict, filename)
      end

      def load(filename)
        @agent.load_state_dict(Torch.load(filename))
        @agent.policy_model.eval
        @agent.value_model.eval
      end

      def eval
        @agent.eval
      end

      def predict(state, deterministic: true)
        # A ternary, not `argument = ... unless state.is_a?(Torch::Tensor)`:
        # that modifier form assigns nil whenever the condition holds, so a
        # state that was already a Tensor reached the policy as nil.
        argument = state.is_a?(Torch::Tensor) ? state : Torch.tensor(state, dtype: :float32)
        logits = Torch.no_grad { @agent.policy_model.call(argument) }
        @environment.actions[CategoricalDistribution.new(logits: logits).choose(deterministic: deterministic)]
      end

      def clip_grad_norm_(parameters, max_norm, norm_type: 2)
        # Calculate the total norm (L2 norm by default) of all gradients
        total_norm = 0
        parameters.each do |param|
          unless param.grad.nil?
            param_norm = param.grad.data.norm(norm_type)
            total_norm += param_norm.item**norm_type
          end
        end
        total_norm **= (1.0 / norm_type)
        # Scale gradients if the total norm exceeds the maximum allowed norm
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1
          parameters.each do |param|
            param.grad&.data&.mul!(clip_coef)
          end
        end

        total_norm
      end

      def calculate_total_grad_norm(parameters, norm_type: 2)
        total_norm = 0.0
        parameters.each do |param|
          unless param.grad.nil?
            param_norm = param.grad.data.norm(norm_type)
            total_norm += param_norm.item**norm_type
          end
        end
        total_norm**(1.0 / norm_type)
      end

      # Split a (shuffled) list of rollout indices into consecutive,
      # non-overlapping minibatches of @minibatch_size (the last one may be
      # smaller), covering every index exactly once per epoch. This used to
      # be an inline `b_inds[start..start + @minibatch_size]` -- an inclusive
      # range, so each minibatch had one sample too many and consecutive
      # minibatches shared one.
      def minibatches(indices)
        indices.each_slice(@minibatch_size).to_a
      end

      def build_rollout_buffers(num_steps)
        {
          obs: Torch.zeros(num_steps, @environment.state_size),
          actions: Torch.zeros(num_steps),
          logprobs: Torch.zeros(num_steps, dtype: :float32),
          rewards: Torch.zeros(num_steps),
          dones: Torch.zeros(num_steps, dtype: :float32),
          values: Torch.zeros(num_steps)
        }
      end

      def compute_gae(rewards, values, dones, next_value, next_done)
        rewards_a = rewards.to_a.map(&:to_f)
        values_a = values.to_a.map(&:to_f)
        dones_a = dones.to_a.map(&:to_f)
        next_value_f = next_value.is_a?(Torch::Tensor) ? next_value.item.to_f : next_value.to_f

        advantages = Array.new(rewards_a.size, 0.0)
        lastgaelam = 0.0

        (rewards_a.size - 1).downto(0).each do |t|
          if t == rewards_a.size - 1
            next_nonterminal = next_done ? 0.0 : 1.0
            next_values = next_value_f
          else
            next_nonterminal = 1.0 - dones_a[t + 1]
            next_values = values_a[t + 1]
          end

          delta = rewards_a[t] + @discount_factor * next_values * next_nonterminal - values_a[t]
          lastgaelam = delta + @discount_factor * @gaelam * next_nonterminal * lastgaelam
          advantages[t] = lastgaelam
        end

        returns = advantages.each_with_index.map { |adv, i| adv + values_a[i] }
        [Torch.tensor(advantages, dtype: :float32), Torch.tensor(returns, dtype: :float32)]
      end

      def train(episodes:, steps_per_episode:, **_kwargs)
        # Initialize the experience buffer, tensors that will store the data
        num_steps = steps_per_episode
        buffers = build_rollout_buffers(num_steps)
        obs = buffers[:obs]
        actions = buffers[:actions]
        logprobs = buffers[:logprobs]
        rewards = buffers[:rewards]
        dones = buffers[:dones]
        values = buffers[:values]

        global_step = 0
        next_obs = @environment.reset
        next_obs = Torch.tensor(next_obs.map!(&:to_f))
        next_done = false
        episode_lenth = 0
        # Loop over the episodes

        1.upto(episodes) do |episode_number|
          progress = episode_number.to_f / episodes * 100
          print "\rTraining: #{progress.round(2)}%" if episode_number % 10 == 0
          # Anneal the learning rate
          fract = 1.0 - (episode_number - 1) / episodes
          lrnow = @learning_rate * fract
          @optimizer.param_groups[0][:lr] = lrnow
          episode_reward = 0

          num_steps.times do |step|
            global_step += 1
            obs[step] = next_obs
            dones[step] = next_done ? 1.0 : 0.0

            action, prob, value = nil

            Torch.no_grad do
              action, prob, _, value = @agent.get_action_and_value(next_obs)
              values[step] = value.flatten
            end

            episode_lenth += 1

            actions[step] = action
            logprobs[step] = prob

            next_obs, reward, next_done = @environment.step(action.to_i)
            rewards[step] = reward
            episode_reward += reward
            if next_done == true || step == num_steps - 1
              @logs[:episode_reward] << episode_reward
              @logs[:episode_length] << episode_lenth
              # warn "Episode: #{episode_number} Reward: #{episode_reward} Length: #{episode_lenth}"
              episode_lenth = 0
              next_obs = @environment.reset
              next_obs.map!(&:to_f)
            end
            next_obs = Torch.tensor(next_obs)
            # TODO log additional information here, e.g. rewards, etc.
            # everything we would to log somewhere
          end

          # calculate the advantages
          returns = nil
          advantages = nil

          # Calculate the advantages and returns
          Torch.no_grad do
            next_value = @agent.get_value(next_obs).reshape(-1)[0]
            advantages, returns = compute_gae(rewards, values, dones, next_value, next_done)
          end

          b_obs = obs.reshape(-1, @environment.state_size)
          b_logprobs = logprobs.reshape(-1)
          b_actions = actions.reshape(-1)
          b_returns = returns.reshape(-1)
          b_advantages = advantages.reshape(-1)
          b_values = values.reshape(-1)

          b_inds = (0...num_steps).to_a
          clipfracs = []

          1.upto(@ppo_epochs) do |epoch|
            b_inds.shuffle!
            minibatches(b_inds).each do |mb_inds|
              # Use the entropy tensor as returned: re-wrapping it in
              # Torch.tensor(...), as this used to, copies its values but drops
              # its autograd history, so the entropy bonus below contributed no
              # gradient at all and PPO trained without one.
              _, newlogprob, entropy, newvalue = @agent.get_action_and_value(b_obs[Torch.tensor(mb_inds)], b_actions[Torch.tensor(mb_inds)])
              logratio = newlogprob - b_logprobs[Torch.tensor(mb_inds)]
              ratio = logratio.exp

              # Calculate the clipfrac
              # Not logging them at the moment
              Torch.no_grad do
                ratiom = (ratio - Torch.tensor(1)).abs
                trues = 0
                ratiom.to_a.each do |r|
                  trues += 1 if r.to_f > @clip_param
                end
                meanratio = trues / ratiom.to_a.size
                clipfracs << meanratio
              end

              mb_advantages = b_advantages[Torch.tensor(mb_inds)]
              # if we want to normalize the advantages
              # use the configuration below
              mb_advantages = (mb_advantages - mb_advantages.mean) / (mb_advantages.std + 1e-8)

              pg_loss1 = -mb_advantages * ratio
              pg_loss2 = -mb_advantages * Torch.clamp(ratio, 1.0 - @clip_param, 1.0 + @clip_param)
              pg_loss = Torch.max(pg_loss1, pg_loss2).mean

              # Use v_clip for calculating value loss
              newvalue = newvalue.reshape(-1)
              v_loss_unclipped = (newvalue - b_returns[Torch.tensor(mb_inds)]).pow(2)
              v_clipped = b_values[Torch.tensor(mb_inds)] + Torch.clamp(newvalue - b_values[Torch.tensor(mb_inds)], -@clip_param, @clip_param)
              v_loss_clipped = (v_clipped - b_returns[Torch.tensor(mb_inds)]).pow(2)
              v_loss_max = Torch.max(v_loss_unclipped, v_loss_clipped)
              value_loss = 0.5 * v_loss_max.mean

              entropy_loss = entropy.mean

              # warn "entropy_loss: #{entropy_loss}"
              loss = pg_loss - @entropy_coefficient * entropy_loss + value_loss * 0.5
              @logs[:loss] << loss.item

              # Here another type of loss without entropy
              # loss = pg_loss + value_loss * 0.5

              @optimizer.zero_grad

              # warn "loss: #{loss}"
              loss.backward
              # Gradient clipping calculation
              clip_grad_norm_(@agent.parameters, @clip_param)

              @optimizer.step
            end
          end
        end
      end
    end
  end
end
