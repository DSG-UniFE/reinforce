# frozen_string_literal: true

require_relative '../experience'
require_relative '../categorical_distribution'
require_relative './deep_sets'
require_relative '../dummy_vectorized_environment'
require 'torch'


module Reinforce
    module Algorithms
      class AgentDS < Torch::NN::Module

        attr_accessor :policy_model, :value_model

        def initialize(state_size, num_actions, policy_model=nil, value_model=nil)
          super()
          if policy_model.nil? || value_model.nil?
            @policy_model = EquivariantDeepSet.new(state_size[1])
            @policy_model.train
            @value_model = InvariantDeepSet.new(state_size[1])
            @value_model.train
          else
            @policy_model = policy_model
            @value_model = value_model
            [@policy_model, @value_model].each(&:train)
          end
        end

        def get_value(x)
          @value_model.call(x)
        end 

        def get_action(x, mask=nil, deterministic=false)
          logits = @policy_model.call(x)
          unless mask.nil?
            huge_neg = Torch.tensor(-1e8)
            logits = Torch.where(mask, logits, huge_neg)
          end 
          pd = CategoricalDistribution.new(logits: logits)
          if deterministic
            pd.mode
          else
            pd.sample
          end
        end 

        def get_action_and_value(x, action=nil, mask=nil)
          logits = @policy_model.call(x)
          unless mask.nil?
            huge_neg = Torch.tensor(-1e8) 
            logits = Torch.where(mask, logits, huge_neg)
            #warn "logits: #{logits}"
          end
          pd = CategoricalDistribution.new(logits: logits)
          action = pd.sample if action.nil?
          value = @value_model.call(x)
          #warn "action: #{action} value: #{value}"
          #warn "x #{x} value: #{value}"
          logprob = pd.log_probability(action)
          entropy = pd.entropy
          [action, logprob, entropy, value]
        end
      end


      class PPODS 

        attr_reader :logs
        attr_accessor :agent,:optimizer
        def initialize(environment, learning_rate, policy=nil,
            value=nil, clip_param = 0.2, ppo_epochs = 4,
            minibatch_size = 128, discount_factor = 0.99)
          @environment = environment
          @agent = AgentDS.new(environment.state_size, environment.actions.size, policy, value)
          @gaelam = 0.97
          @clip_param = clip_param
          @ppo_epochs = ppo_epochs
          @minibatch_size = minibatch_size
          @discount_factor = discount_factor
          @learning_rate = learning_rate
          # Create the optimizer
          @logs = {loss: [], episode_reward: [], episode_length: []}
          @optimizer = Torch::Optim::Adam.new(@agent.parameters,
                                              lr: @learning_rate, eps: 1e-5)
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

        def predict(state, mask=nil)
          Torch.no_grad do
            argument = Torch.tensor(state, dtype: :float32) unless state.is_a?(Torch::Tensor)
            mask = Torch.tensor(mask, dtype: :bool) unless mask.nil?
            @agent.get_action(argument, mask, false)
          end
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
          total_norm**(1.0 / norm_type)
        end

        def train(num_episodes, batch_size)
          # Initialize the experience buffer, tensors that will store the data
          num_steps = batch_size
          # Create a tensor of array to store the observations
          # Each observation has a shape of (6, 3) (6 is the number of environments and 3 is the state size)
          obs = Torch.zeros(num_steps, @environment.state_size[0], @environment.state_size[1])
          actions = Torch.zeros(num_steps, 1)
          logprobs = Torch.zeros(num_steps, 1,dtype: :float32)
          rewards = Torch.zeros(num_steps, 1)
          dones = Torch.zeros(num_steps, 1)
          values = Torch.zeros(num_steps, 1)
          warn "#{@environment.state_size[0]}"
          masks = Torch.zeros(num_steps, 1, @environment.action_masks.length, dtype: :bool)

          global_step = 0
          next_obs = @environment.reset
          next_obs = Torch.tensor(next_obs)#.map!(&:to_f))
          next_done = Torch.zeros(1) # 1 is th number of environments
          next_mask = Torch.tensor(@environment.action_masks)
          episode_lenth = 0
          episode_reward = 0
          1.upto(num_episodes) do |episode_number|
            progress = episode_number.to_f / num_episodes * 100
            print "\rTraining: #{progress.round(2)}%" if episode_number % 10 == 0
            # Anneal the learning rate
            fract = 1.0 - (episode_number - 1) / num_episodes
            lrnow = @learning_rate * fract
            @optimizer.param_groups[0][:lr] = lrnow

            batch_size.times do |step|

              global_step += 1
              episode_lenth += 1
              obs[step] = next_obs
              dones[step] = next_done
              masks[step] = next_mask

              action, prob, value = nil

              Torch.no_grad do
                action, prob, _, value = @agent.get_action_and_value(next_obs)
                values[step] = value.flatten
              end


              actions[step] = action
              logprobs[step] = prob

              next_obs, reward, next_done = @environment.step(action.to_i)
              rewards[step] = Torch.tensor(reward).view(-1)
              episode_reward += (reward.sum / reward.length)

              if next_done == [true]
                @logs[:episode_reward] << episode_reward
                @logs[:episode_length] << episode_lenth
                #warn "Episode: #{episode_number} Reward: #{episode_reward} Length: #{episode_lenth}"
                episode_lenth = 0
                episode_reward = 0
                next_obs = @environment.reset
                next_mask = Torch.tensor(@environment.action_masks)
              end
              next_obs = Torch.tensor(next_obs)
              next_done = Torch.tensor(next_done)
              next_mask = Torch.tensor(@environment.action_masks)
            end


            # calculate the advantages
            returns = nil
            advantages = nil

            # Calculate the advantages and returns
            Torch.no_grad do
              next_value = @agent.get_value(next_obs).reshape(1, -1)
              advantages = Torch.zeros_like(rewards)
              lastgaelam = 0
              next_done_f = next_done == [true] ? 1.0 : -1.0

              (num_steps - 1).downto(0).each do |t|
                if t == num_steps - 1
                  nextnonterminal = 1.0 - next_done_f
                  nextvalues = next_value
                else
                  nextnonterminal = 1 - dones[t + 1]
                  nextvalues = values[t + 1]
                end
                delta = rewards[t] + @discount_factor * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + @discount_factor * @gaelam * nextnonterminal * lastgaelam
              end
              returns = advantages + values
            end

            # We need to reshape the observations to be able to pass them to the model
            # We want to feed the model with a batch of observations
            # The model expects a batch of observations with the shape (batch_size, state_size)
            # currently obs is a tensor with the shape (num_steps, state_size, 1)
            b_obs = obs.reshape([-1,] + @environment.state_size)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape(-1, 1)
            b_masks = masks.reshape(-1, @environment.action_masks.length)
            b_returns = returns.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_values = values.reshape(-1)

            b_inds = (0...num_steps).to_a 

            1.upto(@ppo_epochs) do |epoch|
              b_inds.shuffle! 
              (0..(batch_size -1)).step(@minibatch_size) do |start|
                end_s = start + @minibatch_size 
                mb_inds = b_inds[start..end_s]
                _, newlogprob, entropy, newvalue = @agent.get_action_and_value(b_obs[Torch.tensor(mb_inds)], b_actions[Torch.tensor(mb_inds)], b_masks[Torch.tensor(mb_inds)])
                entropy = Torch.tensor(entropy)
                #warn "b_obs: #{b_obs[Torch.tensor(mb_inds)]}" if entropy.to_f.nan?
                #warn "b_actions: #{b_actions[Torch.tensor(mb_inds)]}" if entropy.to_f.nan?

                #warn "newlogprob: #{newlogprob} @entropy: #{entropy} newvalue: #{newvalue}"
                logratio = newlogprob - b_logprobs[Torch.tensor(mb_inds)]
                ratio = logratio.exp

                mb_advantages = b_advantages[Torch.tensor(mb_inds)]
                # if we want to normalize the advantages
                # use the configuration below
                mb_advantages = (mb_advantages - mb_advantages.mean) / (mb_advantages.std + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * Torch.clamp(ratio, 1.0 - @clip_param, 1.0 + @clip_param)
                pg_loss = Torch.max(pg_loss1, pg_loss2).mean

                # Use v_clip for calculating value loss
                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - b_returns[Torch.tensor(mb_inds)]).pow(2)
                v_clipped = b_values[Torch.tensor(mb_inds)] + Torch.clamp(newvalue - b_values[Torch.tensor(mb_inds)], -@clip_param, @clip_param)
                v_loss_clipped = (v_clipped - b_returns[Torch.tensor(mb_inds)]).pow(2)
                v_loss_max = Torch.max(v_loss_unclipped, v_loss_clipped)
                value_loss = 0.5 * v_loss_max.mean

                entropy_loss = entropy.mean

                #warn "entropy_loss: #{entropy_loss}"
                loss = pg_loss - 0.01 * entropy_loss + value_loss * 0.5
                @logs[:loss] << loss.item
                #warn "loss: #{loss} entropy_loss: #{entropy_loss} value_loss: #{value_loss} pg_loss: #{pg_loss}"

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
