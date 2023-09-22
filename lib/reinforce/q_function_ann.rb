# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi, Filippo Poltronieri

require 'torch'
require 'forwardable'

module Reinforce
  # input to the network is the current state
  # output of the network is the log probabilities of each action
  class QFunctionANN
    extend Forwardable
    def_delegators :@architecture, :apply, :parameters, :state_dict, :load_state_dict

    def initialize(state_size, num_actions, learning_rate, discount_factor)
      @num_actions = num_actions
      @architecture = Torch::NN::Sequential.new(
        Torch::NN::Linear.new(state_size, 512),
        Torch::NN::ReLU.new,
        Torch::NN::Linear.new(512, 512),
        Torch::NN::ReLU.new,
        Torch::NN::Linear.new(512, num_actions)
      )
      @architecture.train # Enable training mode
      # Create the optimizer
      @optimizer = Torch::Optim::Adam.new(@architecture.parameters, lr: learning_rate)
      @discount_factor = discount_factor
    end

    def forward(state)
      argument = if state.is_a?(Torch::Tensor)
                   state
                 else
                   Torch::Tensor.new(state)
                 end
      @architecture.forward(argument)
    end

    def random_action(_state)
      rand(@num_actions)
    end

    def update(experience)
      next_actions = experience[:next_states].map do |next_state|
        # Need to tell Torch not to track the gradient for these operations.
        # See L. Graesser, W.L. Keng, "Foundations of Deep Reinforcement
        # Learning", Section 3.5.2, page 70.
        Torch.no_grad { forward(next_state).argmax.to_i }
      end
      target_actions = next_actions
                       .zip(experience[:rewards], experience[:dones]).map do |next_action, reward, done|
        if done
          reward
        else
          reward + @discount_factor * next_action
        end
      end
      criterion = Torch::NN::MSELoss.new
      @optimizer.zero_grad
      experience_actions = experience[:actions]
      warn "target_actions: #{target_actions.inspect}"
      warn "experience_actions: #{experience_actions.inspect}"
      loss = criterion.call(Torch::Tensor.new(target_actions), Torch::Tensor.new(experience_actions))
      @optimizer.step(proc { loss })
    end

    def soft_update(q_network, tau)
      # Check validity for tau interpolation parameter
      tau = tau.to_f unless tau.is_a?(Float)
      raise ArgumentError 'tau needs to be a real in the (0,1) range' if tau < 0.0 || tau > 1.0

      # Load the parameters of the present network
      my_params = @architecture.state_dict.transform_values { |v| v.data.clone }

      # Load the parameters of the other network
      other_q_params = q_network.state_dict.transform_values { |v| v.data.clone }

      # Obtain a new set of parameters through interpolation
      new_params = {}
      my_params.each do |k, v|
        new_params[k] = v * (1 - tau) + other_q_params[k] * tau
      end

      # Load the new parameters
      @architecture.load_state_dict(new_params)
    end
  end

  def save(path)
    Torch.save(@architecture.state_dict, path)
  end

  def load(path)
    @architecture.load_state_dict(Torch.load(path))
    @architecture.eval
  end
end
