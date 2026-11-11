# frozen_string_literal: true

require_relative "../experience"
require_relative "../categorical_distribution"
require_relative "../networks"
require_relative "../running_mean_std"
require_relative "../hyperparameters"
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

      # A fresh value head shaped like the default one (used by PPO for its
      # separate intrinsic-reward value estimate).
      def build_value_head(state_size, network_config: DEFAULT_NETWORK_CONFIG)
        build_head(state_size, 1, network_config, output_std: 1.0).to(Reinforce::Device.resolve(network_config.device))
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

      # Added to the return variance before taking its square root when
      # normalizing rewards, as in Gymnasium's NormalizeReward.
      REWARD_NORMALIZATION_EPSILON = 1e-8

      # Hyperparameters (see Reinforce::Hyperparameters).
      #
      # learning_rate: initial Adam learning rate, annealed linearly to zero
      #   over training.
      # clip_param: PPO's probability-ratio clipping range (also used to
      #   clip value-function updates).
      # ppo_epochs / minibatch_size: optimization passes over each rollout,
      #   and samples per gradient step.
      # gae_lambda: Generalized Advantage Estimation's lambda.
      # max_grad_norm: gradient-norm clipping threshold.
      # normalize_rewards: see the comment in #initialize.
      # network: the shape of the default policy and value networks (only
      #   used when `policy_model:`/`value_model:` aren't given).
      #
      # With an `intrinsic_reward:` (see #initialize):
      # intrinsic_value_head: learn a separate value estimate for the
      #   intrinsic reward, as in RND (Burda et al., 2018), instead of adding
      #   the bonus to the extrinsic reward (false). On a sparse one-hot
      #   15x15 four-rooms GridWorld (30,000 steps, 5 seeds), a count bonus
      #   added to the reward explored 98-100% of the grid but solved 0
      #   seeds; with the separate head it solved all 5 (optimal path). The intrinsic stream is
      #   non-episodic (novelty doesn't reset with the episode), discounted by
      #   intrinsic_discount_factor and normalized on its own; the policy
      #   follows extrinsic_advantage_coefficient * A_ext +
      #   intrinsic_advantage_coefficient * A_int.
      # intrinsic_advantage_decay: decay the intrinsic coefficient linearly
      #   to zero over training (alongside the learning rate), shifting from
      #   exploration to exploitation.
      Config = Data.define(
        :learning_rate, :clip_param, :ppo_epochs, :minibatch_size, :discount_factor, :gae_lambda,
        :entropy_coefficient, :max_grad_norm, :normalize_rewards, :network,
        :intrinsic_value_head, :extrinsic_advantage_coefficient, :intrinsic_advantage_coefficient,
        :intrinsic_discount_factor, :intrinsic_advantage_decay
      ) do
        def initialize(learning_rate: 2.5e-4, clip_param: 0.2, ppo_epochs: 10, minibatch_size: 32, discount_factor: 0.99,
          gae_lambda: 0.97, entropy_coefficient: 0.01, max_grad_norm: 0.2, normalize_rewards: true,
          network: ActorCriticNetwork::DEFAULT_NETWORK_CONFIG,
          intrinsic_value_head: true, extrinsic_advantage_coefficient: 2.0, intrinsic_advantage_coefficient: 1.0,
          intrinsic_discount_factor: 0.99, intrinsic_advantage_decay: false)
          super
        end
      end

      attr_reader :metrics, :config

      alias_method :logs, :metrics
      attr_accessor :agent, :optimizer

      # @parameter intrinsic_reward [#observe, #call, #train, nil] an
      #   exploration bonus added to each reward as rollouts are collected
      #   (see Reinforce::IntrinsicRewards).
      def initialize(environment, config = nil, policy_model: nil, value_model: nil, intrinsic_reward: nil, **hyperparameters)
        @config = Reinforce::Hyperparameters.resolve(Config, config, hyperparameters)
        @environment = environment
        @intrinsic_reward = intrinsic_reward
        # Reward normalization (Gymnasium's NormalizeReward): divide each
        # reward by a running estimate of the standard deviation of the
        # discounted return, so the critic's targets stay near unit scale
        # whatever the environment's reward scale. Without it, returns in
        # the hundreds (as in GridWorld) leave the critic far behind them,
        # and its large value-loss gradients -- clipped jointly with the
        # policy's -- shrink every policy update. Rewards are only scaled,
        # never shifted, so their sign and the optimal policy are unchanged.
        # The statistics persist across #train calls, but not across
        # #save/#load. `episode_reward` metrics always report raw rewards.
        @normalize_rewards = @config.normalize_rewards
        @return_rms = Reinforce::RunningMeanStd.new
        @discounted_return = 0.0
        @entropy_coefficient = @config.entropy_coefficient
        @max_grad_norm = @config.max_grad_norm
        @agent = ActorCriticNetwork.new(
          environment.state_size, environment.actions.size, policy_model, value_model, network_config: @config.network
        )
        @gaelam = @config.gae_lambda
        @clip_param = @config.clip_param
        @ppo_epochs = @config.ppo_epochs
        @minibatch_size = @config.minibatch_size
        @discount_factor = @config.discount_factor
        @learning_rate = @config.learning_rate
        @two_heads = !@intrinsic_reward.nil? && @config.intrinsic_value_head
        if @two_heads
          @intrinsic_value_model = @agent.build_value_head(environment.state_size, network_config: @config.network)
          @intrinsic_return_rms = Reinforce::RunningMeanStd.new
          @intrinsic_discounted_return = 0.0
        end
        # Create the optimizer
        @metrics = Reinforce::Metrics::Recorder.new
        @optimizer = Torch::Optim::Adam.new(trainable_parameters, lr: @learning_rate, eps: 1e-5)
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

      # Convert an environment observation to the float tensor the networks
      # expect. Non-mutating: this used to `map!(&:to_f)` the array returned
      # by #reset in place, which silently rewrote the environment's own
      # state (and only converted reset observations, so an environment
      # whose #step returns Integer observations crashed the policy).
      def observation_tensor(observation)
        Torch.tensor(observation.map(&:to_f), dtype: :float32)
      end

      # Scale one reward by the running standard deviation of the discounted
      # return it contributes to (see the comment in #initialize). Unlike
      # Gymnasium, which multiplies the accumulator by (1 - terminated) as
      # part of the terminal step's own update, the accumulator is reset
      # *after* an episode ends (in #train), so a terminal reward is still
      # added to its own episode's return.
      def normalize_reward(reward)
        @discounted_return = @discounted_return * @discount_factor + reward
        @return_rms.update(@discounted_return)
        reward / Math.sqrt(@return_rms.var + REWARD_NORMALIZATION_EPSILON)
      end

      # Intrinsic rewards are scaled by the running std of their own
      # non-episodic discounted return (as in RND), so the bonus's scale
      # doesn't depend on its coefficient or on how familiar states are.
      def normalize_intrinsic_reward(bonus)
        @intrinsic_discounted_return = @intrinsic_discounted_return * @config.intrinsic_discount_factor + bonus
        @intrinsic_return_rms.update(@intrinsic_discounted_return)
        bonus / Math.sqrt(@intrinsic_return_rms.var + REWARD_NORMALIZATION_EPSILON)
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

      # The advantage the policy follows in two-head mode. `remaining` is the
      # fraction of training left (1.0 for the first rollout), which scales
      # the intrinsic weight when intrinsic_advantage_decay is on.
      def combine_advantages(extrinsic, intrinsic, remaining)
        intrinsic_coefficient = @config.intrinsic_advantage_coefficient
        intrinsic_coefficient *= remaining if @config.intrinsic_advantage_decay
        @config.extrinsic_advantage_coefficient * extrinsic + intrinsic_coefficient * intrinsic
      end

      # Every parameter the optimizer updates (and gradient clipping covers).
      def trainable_parameters
        models = [@agent.policy_model, @agent.value_model]
        models << @intrinsic_value_model if @intrinsic_value_model
        models.flat_map(&:parameters)
      end

      def compute_gae(rewards, values, dones, next_value, next_done, discount_factor: @discount_factor)
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

          delta = rewards_a[t] + discount_factor * next_values * next_nonterminal - values_a[t]
          lastgaelam = delta + discount_factor * @gaelam * next_nonterminal * lastgaelam
          advantages[t] = lastgaelam
        end

        returns = advantages.each_with_index.map { |adv, i| adv + values_a[i] }
        [Torch.tensor(advantages, dtype: :float32), Torch.tensor(returns, dtype: :float32)]
      end

      # @param progress_output [IO, nil] where to print training progress
      # (e.g. `$stdout`); nothing is printed when nil (the default)
      def train(episodes:, steps_per_episode:, progress_output: nil, **_kwargs)
        # Initialize the experience buffer, tensors that will store the data
        num_steps = steps_per_episode
        buffers = build_rollout_buffers(num_steps)
        obs = buffers[:obs]
        actions = buffers[:actions]
        logprobs = buffers[:logprobs]
        rewards = buffers[:rewards]
        dones = buffers[:dones]
        values = buffers[:values]
        intrinsic_rewards = Torch.zeros(num_steps)
        intrinsic_values = Torch.zeros(num_steps)

        global_step = 0
        next_obs = observation_tensor(@environment.reset)
        next_done = false
        episode_lenth = 0
        episode_reward = 0
        episode_bonus = 0.0
        # Loop over the episodes

        1.upto(episodes) do |episode_number|
          progress = episode_number.to_f / episodes * 100
          progress_output&.print "\rTraining: #{progress.round(2)}%" if episode_number % 10 == 0
          # Anneal the learning rate linearly towards zero. fdiv, not `/`:
          # with Integer division this was 0 for every rollout, so the
          # learning rate never annealed at all.
          fract = 1.0 - (episode_number - 1).fdiv(episodes)
          lrnow = @learning_rate * fract
          @optimizer.param_groups[0][:lr] = lrnow
          bootstrap_obs = nil
          rollout_next_states = []

          num_steps.times do |step|
            global_step += 1
            obs[step] = next_obs
            dones[step] = next_done ? 1.0 : 0.0

            action, prob, value = nil

            Torch.no_grad do
              action, prob, _, value = @agent.get_action_and_value(next_obs)
              values[step] = value.flatten
              intrinsic_values[step] = @intrinsic_value_model.call(next_obs).flatten if @two_heads
            end

            episode_lenth += 1

            actions[step] = action
            logprobs[step] = prob

            next_obs, reward, next_done = @environment.step(action.to_i)
            training_reward = reward
            if @intrinsic_reward
              @intrinsic_reward.observe(next_obs)
              rollout_next_states << next_obs
              bonus = @intrinsic_reward.call([next_obs]).first
              episode_bonus += bonus
              if @two_heads
                intrinsic_rewards[step] = normalize_intrinsic_reward(bonus)
              else
                training_reward += bonus
              end
            end
            rewards[step] = @normalize_rewards ? normalize_reward(training_reward) : training_reward
            episode_reward += reward
            if next_done == true || step == num_steps - 1
              @metrics.record(:episode_reward, episode_reward)
              @metrics.record(:episode_length, episode_lenth)
              @metrics.record(:intrinsic_reward, episode_bonus) if @intrinsic_reward
              episode_reward = 0
              episode_bonus = 0.0
              episode_lenth = 0
              @discounted_return = 0.0
              # An episode cut short by the end of the rollout did not
              # terminate: its last transition must bootstrap from the
              # state it actually reached, not from the reset state below.
              bootstrap_obs = observation_tensor(next_obs) unless next_done
              next_obs = @environment.reset
            end
            next_obs = observation_tensor(next_obs)
          end
          @intrinsic_reward&.train(rollout_next_states)

          # calculate the advantages
          returns = nil
          advantages = nil
          intrinsic_returns = nil

          # Calculate the advantages and returns
          Torch.no_grad do
            next_value = @agent.get_value(bootstrap_obs || next_obs).reshape(-1)[0]
            advantages, returns = compute_gae(rewards, values, dones, next_value, next_done)
            if @two_heads
              # Non-episodic: no dones, and the stream simply continues into
              # whatever state comes next, reset or not.
              next_intrinsic_value = @intrinsic_value_model.call(next_obs).reshape(-1)[0]
              intrinsic_advantages, intrinsic_returns = compute_gae(
                intrinsic_rewards, intrinsic_values, Torch.zeros(num_steps), next_intrinsic_value, false,
                discount_factor: @config.intrinsic_discount_factor
              )
              advantages = combine_advantages(advantages, intrinsic_advantages, fract)
            end
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
              if @two_heads
                new_intrinsic_value = @intrinsic_value_model.call(b_obs[Torch.tensor(mb_inds)]).reshape(-1)
                value_loss += 0.5 * (new_intrinsic_value - intrinsic_returns[Torch.tensor(mb_inds)]).pow(2).mean
              end

              entropy_loss = entropy.mean

              # warn "entropy_loss: #{entropy_loss}"
              loss = pg_loss - @entropy_coefficient * entropy_loss + value_loss * 0.5
              @metrics.record(:loss, loss.item)

              # Here another type of loss without entropy
              # loss = pg_loss + value_loss * 0.5

              @optimizer.zero_grad

              # warn "loss: #{loss}"
              loss.backward
              # Gradient clipping calculation
              clip_grad_norm_(trainable_parameters, @max_grad_norm)

              @optimizer.step
            end
          end
        end
      end
    end
  end
end
