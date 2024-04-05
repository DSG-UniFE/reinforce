# frozen_string_literal: true


require 'torch'

# Let's define the class for creating the DeepSet version of the algorithm, e.g., PPO

module Reinforce
  module Algorithms

  # Create a class that performs the same operation of a DummyVectorizedEnvironment
  class DummyVectorizedEnvironment
    def initialize(environment, num_envs)
      @environment = environment
      @num_envs = num_envs
      @states = Array.new(num_envs) { @environment.reset }
    end

    def state_size
      #warn "environment.state_size: #{@environment.state_size}"
      @environment.state_size
    end

    def step(actions)
      #warn "Applying actions to vectorized environment---> actions: #{actions}"
      actions = [actions] if actions.is_a?(Integer)
      nobs = []
      nrews = []
      ndones = []
      @states.zip(actions).map do |state, action|
        #warn "state: #{state}, action: #{action}"
        nob, nrew, ndone = @environment.step(action) 
        nobs << nob
        nrews << nrew
        ndones << ndone
      end
      [nobs, nrews, ndones]
    end 

    def reset
      @states = Array.new(@num_envs) { @environment.reset }
    end

    def actions
      @environment.actions
    end
  end 

  # torch.rb does not define a binding for the ELU activation function in the Torch::NN module.
  # We can define it ourselves:
  # Elu activation function
  class ELU < Torch::NN::Module 
    def initialize(alpha = 1.0, inplace = false)
      super()
      @alpha = alpha
      @inplace = inplace
    end

    def forward(x)
      # x: input tensor
      # implement the mathematically defined ELU activation function
      # return: output tensor
      @alpha * Torch.exp(x) - 1
      #Torch.where(x > 0, x, @alpha * (Torch.exp(x) - 1))
    end
  end 

  
  class EquivariantLayer < Torch::NN::Module
    def initialize(input_size, output_size)
      #warn "EquivariantLayer: input_size: #{input_size}, output_size: #{output_size}"
      super()
      @gamma = Torch::NN::Linear.new(input_size, output_size, bias: false)
      @lambda = Torch::NN::Linear.new(input_size, output_size, bias: false)
    end

    def forward(x)
      # x: (batch_:size, n_elements, in_channels)
      # return: (batch_size, n_elements)
      xm,_ = Torch.max(x, dim: 1, keepdim: true)
      #warn "xm: #{xm.shape} xm: #{xm}"
      #warn "lambda - gamma: #{(@lambda.call(x) - @gamma.call(xm)).shape}"
      #warn "lambda.call(x) #{@lambda.call(x)}"
      @lambda.call(x) - @gamma.call(xm) 
    end
  end

  class EquivariantDeepSet < Torch::NN::Module 
    def initialize(input_size, hidden_size = 64)
      warn "EquivariantDeepSet input_size: #{input_size}, hidden_size: #{hidden_size}"
      super()
      @net = Torch::NN::Sequential.new(
      EquivariantLayer.new(input_size, hidden_size),
      Torch::NN::ReLU.new,
      EquivariantLayer.new(hidden_size, hidden_size),
      Torch::NN::ReLU.new,
      EquivariantLayer.new(hidden_size, 1)
      )
    end

    def forward(x)
      # x: (batch_size, n_elements, in_channels)
      # return: (batch_size, 1)
      #f = Torch.squeeze(@net.call(x), dim: -1)
      #warn "x in forward: #{x.shape}"
      x = @net.call(x)
      Torch.squeeze(x, dim: -1)
    end
  end

  class InvariantDeepSet < Torch::NN::Module
    def initialize(input_size, hidden_size = 64)
      super()
      warn "InvariantDeepSet input_size: #{input_size}, hidden_size: #{hidden_size}"
      @psi = Torch::NN::Sequential.new(
        EquivariantLayer.new(input_size, hidden_size),
        Torch::NN::ReLU.new,

        EquivariantLayer.new(hidden_size, hidden_size),
        Torch::NN::ReLU.new,
        EquivariantLayer.new(hidden_size, hidden_size)
      ) 
      @rho = Torch::NN::Sequential.new(
        Torch::NN::Linear.new(hidden_size, hidden_size),
        Torch::NN::ReLU.new,
        Torch::NN::Linear.new(hidden_size, 1)
      )
    end

    def forward(x)
      # x: (batch_size, n_elements, in_channels)
      # return: (batch_size, n_elements)
      #warn "x in forward: #{x}"
      x = Torch.mean(@psi.call(x), dim: 1)
      #warn "Invariant: x in forward: #{x}"
      return Torch.squeeze(@rho.call(x), dim: -1)
    end

  end 

  end

end

