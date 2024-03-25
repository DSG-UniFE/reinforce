# frozen_string_literal: true

require_relative '../experience'
require_relative '../categorical_distribution'
require 'torch'

module Reinforce
    module Algorithms
      class Agent < Torch::NN::Module

        attr_reader :policy_model, :value_model

        def initialize(state_size, num_actions, policy_model=nil, value_model=nil)
          super()
          if policy_model.nil? || value_model.nil?
              @policy_model = Torch::NN::Sequential.new(
                  layer_init(Torch::NN::Linear.new(state_size, 64)),
                  Torch::NN::Tanh.new,
                  layer_init(Torch::NN::Linear.new(64, 64)),
                  Torch::NN::Tanh.new,
                  layer_init(Torch::NN::Linear.new(64, num_actions), 0.01))
              @policy_model.train
              @value_model = Torch::NN::Sequential.new(
                  layer_init(Torch::NN::Linear.new(state_size, 64)),
                  Torch::NN::Tanh.new,
                  layer_init(Torch::NN::Linear.new(64, 64)),
                  Torch::NN::Tanh.new,
                  layer_init(Torch::NN::Linear.new(64, 1), 1.0))
              @value_model.train
          else 
            @policy_model = policy_model
            @value_model = value_model
            [@policy_model, @value_model].each(&:train)
          end
        end

        # from cleanrl
        def layer_init(layer, std = Math.sqrt(2), bias_const=0.0)
            Torch::NN::Init.orthogonal!(layer.weight, gain: std)
            Torch::NN::Init.constant!(layer.bias, bias_const)
            return layer
        end

        def get_value(x)
          @value_model.call(x)
        end

        def get_action_and_value(x, action=nil)
          logits = @policy_model.call(x)
          #warn "logits: #{logits}"
          pd = CategoricalDistribution.new(logits: logits)
          action = pd.sample if action.nil?
          value = @value_model.call(x)
          logprob = pd.log_probability(action)
          entropy = pd.entropy
          [action, logprob, entropy, value]
        end
      end


      class PPO 

          attr_reader :logs
          attr_accessor :agent,:optimizer
          def initialize(environment, learning_rate, policy=nil, value=nil, clip_param = 0.2, ppo_epochs = 10, minibatch_size = 32, discount_factor = 0.99)
            @environment = environment
            @agent = Agent.new(environment.state_size, environment.actions.size, policy, value)
            @gaelam = 0.97
            @clip_param = clip_param
            @ppo_epochs = ppo_epochs
            @minibatch_size = minibatch_size
            @discount_factor = discount_factor
            @learning_rate = learning_rate
            # Create the optimizer
            @logs = {loss: [], episode_reward: []}
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

          def eval()
              @agent.eval
          end

          def predict(state)
            argument = Torch.tensor(state, dtype: :float32) unless state.is_a?(Torch::Tensor)
            logits = Torch.no_grad { @agent.policy_model.call(argument) }
            pd = CategoricalDistribution.new(logits: logits)
            pd.sample
          end

          def clip_grad_norm_(parameters, max_norm, norm_type: 2)
            # Calculate the total norm (L2 norm by default) of all gradients
            total_norm = 0
            parameters.each do |param|
              unless param.grad.nil?
                param_norm = param.grad.data.norm(norm_type)
                total_norm += param_norm.item ** norm_type
              end
            end
            total_norm = total_norm ** (1.0 / norm_type)
            # Scale gradients if the total norm exceeds the maximum allowed norm
            clip_coef = max_norm / (total_norm + 1e-6)
            if clip_coef < 1
              parameters.each do |param|
                unless param.grad.nil?
                  param.grad.data.mul!(clip_coef)
                end
              end
            end
            
            return total_norm
          end

          def calculate_total_grad_norm(parameters, norm_type: 2)
            total_norm = 0.0
            parameters.each do |param|
              unless param.grad.nil?
                param_norm = param.grad.data.norm(norm_type)
                total_norm += param_norm.item ** norm_type
              end
            end
            total_norm = total_norm ** (1.0 / norm_type)
            total_norm
          end          

          def train(num_episodes, batch_size)
            # Initialize the experience buffer, tensors that will store the data
            num_steps = batch_size
            obs = Torch.zeros(num_steps + @environment.state_size, @environment.state_size)
            actions = Torch.zeros(num_steps + @environment.actions.size)
            logprobs = Torch.zeros(num_steps, dtype: :float32)
            rewards = Torch.zeros(num_steps)
            dones = Torch.zeros(num_steps)
            values = Torch.zeros(num_steps)

            global_step = 0
            next_obs = @environment.reset
            next_obs = Torch.tensor(next_obs.map!(&:to_f))
            next_done = Torch.zeros(1) # 1 is th number of environments

            # Loop over the episodes

              1.upto(num_episodes) do |episode_number|
                  progress = episode_number.to_f / num_episodes * 100
                  #print "\rTraining: #{progress.round(2)}%"
                  # Anneal the learning rate
                  fract = 1.0 - (episode_number -1) / num_episodes
                  lrnow = @learning_rate * fract
                  @optimizer.param_groups[0][:lr] = lrnow
                  episode_reward = 0

                  batch_size.times do |step|

                    global_step += 1
                    obs[step] = next_obs
                    dones[step] = next_done

                    action, prob, value = nil

                    Torch.no_grad do
                      action, prob, _, value = @agent.get_action_and_value(next_obs)
                      values[step] = value.flatten
                    end

                    actions[step] = action
                    logprobs[step] = prob

                    next_obs, reward, next_done = @environment.step(action.to_i)
                    rewards[step] = reward
                    episode_reward += reward
                    if next_done == true || step == num_steps - 1
                      @logs[:episode_reward] << episode_reward
                      next_obs = @environment.reset
                      next_obs.map!(&:to_f)
                    end
                    next_obs = Torch.tensor(next_obs)
                    next_done = Torch.tensor(next_done)
                    # TODO log additional information here, e.g. rewards, etc.
                    # everything we would to log somewhere 
                  end

                  # calculate the advantages
                  returns = nil
                  advantages = nil

                  # Calculate the advantages and returns
                  Torch.no_grad do
                    next_value = @agent.get_value(next_obs).reshape(1, -1)
                    advantages = Torch.zeros_like(rewards)
                    lastgaelam = 0
                    next_done_f = next_done ? 1.0 : -1.0

                    (num_steps - 1).downto(0).each do |t|
                      if t == num_steps - 1
                        nextnonterminal = 1.0 - next_done_f
                        nextvalues = next_value
                      else
                        nextnonterminal = 1 - (dones[t + 1] ? 1.0 : 0.0)
                        nextvalues = values[t + 1]
                      end
                      delta = rewards[t] + @discount_factor * nextvalues * nextnonterminal - values[t]
                      advantages[t] = lastgaelam = delta + @discount_factor * @gaelam * nextnonterminal * lastgaelam
                    end
                    returns = advantages + values
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
                    (0..(batch_size -1)).step(@minibatch_size) do |start|
                      end_s = start + @minibatch_size 
                      mb_inds = b_inds[start..end_s]
                      _, newlogprob, entropy, newvalue = @agent.get_action_and_value(b_obs[Torch.tensor(mb_inds)], b_actions[Torch.tensor(mb_inds)])
                      entropy = Torch.tensor(entropy)
                      logratio = newlogprob - b_logprobs[Torch.tensor(mb_inds)]
                      ratio = logratio.exp

                      # Calculate the clipfrac
                      # Not logging them at the moment
                      Torch.no_grad do
                        ratiom = (ratio - Torch.tensor(1)).abs()
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
                      pg_loss = Torch.max(pg_loss1, pg_loss2).mean()

                      # Use v_clip for calculating value loss
                      newvalue = newvalue.reshape(-1)
                      v_loss_unclipped = (newvalue - b_returns[Torch.tensor(mb_inds)]).pow(2)
                      v_clipped = b_values[Torch.tensor(mb_inds)] + Torch.clamp(newvalue - b_values[Torch.tensor(mb_inds)], -@clip_param, @clip_param)
                      v_loss_clipped = (v_clipped - b_returns[Torch.tensor(mb_inds)]).pow(2)
                      v_loss_max = Torch.max(v_loss_unclipped, v_loss_clipped)
                      value_loss = 0.5 * v_loss_max.mean()

                      entropy_loss = entropy.mean

                      #warn "entropy_loss: #{entropy_loss}"
                      loss = pg_loss - 0.01 * entropy_loss + value_loss * 0.5
                      @logs[:loss] << loss.item

                      # Here another type of loss without entropy
                      # loss = pg_loss + value_loss * 0.5

                      @optimizer.zero_grad

                      #warn "loss: #{loss}"     
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
