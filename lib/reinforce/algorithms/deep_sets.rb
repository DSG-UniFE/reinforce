# frozen_string_literal: true


require 'torch'

# Let's define the class for creating the DeepSet version of the algorithm, e.g., PPO

module Reinforce
  module Algorithms

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
        Torch::NN::ELU.new,
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
        Torch::NN::ELU.new,
        EquivariantLayer.new(hidden_size, hidden_size),
        Torch::NN::ELU.new,
        EquivariantLayer.new(hidden_size, hidden_size)
      ) 
      @rho = Torch::NN::Sequential.new(
        Torch::NN::Linear.new(hidden_size, hidden_size),
        Torch::NN::ELU.new,
        Torch::NN::Linear.new(hidden_size, 1)
      )
    end

    def forward(x)
      # x: (batch_size, n_elements, in_channels)
      # return: (batch_size, n_elements)
      x = Torch.mean(@psi.call(x), dim: 1)
      return Torch.squeeze(@rho.call(x), dim: -1)
    end

  end 

  end

end

