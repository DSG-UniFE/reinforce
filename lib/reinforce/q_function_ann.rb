# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi, Filippo Poltronieri

require 'torch'
require 'forwardable'
require_relative './categorical_distribution'

module Reinforce
  # input to the network is the current state
  # output of the network is the log probabilities of each action
  class QFunctionANN
    extend Forwardable
    def_delegators :@architecture, :apply, :parameters, :state_dict, :load_state_dict
    attr_reader :optimizer, :architecture

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

    def get_action(state)
      argument = if state.is_a?(Torch::Tensor)
                    state
                  else
                    Torch::Tensor.new(state)
                  end
      logits = Torch.no_grad { forward(argument) }
      CategoricalDistribution.new(logits: logits).sample
    end 


    def random_action(_state)
      rand(@num_actions)
    end

    def update(experience)
      # Need to tell Torch not to track the gradient for these operations.
      # See L. Graesser, W.L. Keng, "Foundations of Deep Reinforcement
      # Learning", Section 3.5.2, page 70.
      next_q_values = Torch.no_grad { forward(experience[:next_state]) }
      # use argmax to select next_actions
      next_actions = next_q_values.argmax(1)
      # compute target actions
      # here we need to create first a tensor of zeros to keep the dimensions and types
      # of the other tensors.
      target_actions = Torch.zeros(experience[:action].size)
      next_actions.zip(experience[:reward], experience[:done]).each_with_index do |(next_action, reward, done), i|
        if done
          target_actions[i] = reward
        else
          # the last part could be next_action, not next_q_values...
          target_actions[i] = reward + @discount_factor * next_q_values[i][next_action]
        end
      end

      # Compute the loss
      # First, we need to extract the q values for the actions taken_q_values
      # from the predicted q values. Here we need a Tensor as well to call backward
      # on loss.
      predicted_q_values = forward(experience[:state])
      taken_q_values = Torch.zeros(experience[:action].size)
      taken_q_values.zip(experience[:action]).each_with_index do |(_, action), i|
        taken_q_values[i] = predicted_q_values[i][action]
      end

      criterion = Torch::NN::MSELoss.new
      @optimizer.zero_grad
      # Some debugging. Comment if not needed.
      #warn "target_actions: #{target_actions.inspect}"
      # Calculate the loss
      loss = criterion.call(taken_q_values, Torch::Tensor.new(target_actions))
      lvalue = loss.item
      # Log the loss
      # warn "Loss: #{loss}"
      # Backpropagate the loss
      loss.backward
      # Update the weights
      @optimizer.step
      lvalue
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

    def save(path)
      Torch.save(@architecture.state_dict, path)
    end

    def load(path)
      @architecture.load_state_dict(Torch.load(path))
      @architecture.eval
    end

  end
end
