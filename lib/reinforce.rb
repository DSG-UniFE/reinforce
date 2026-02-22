# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi.

require_relative "reinforce/version"
require_relative "reinforce/experience"
require_relative "reinforce/categorical_distribution"
require_relative "reinforce/prioritized_experience_replay"
require_relative "reinforce/environments/1d_walk"
require_relative "reinforce/environments/game_2048"
require_relative "reinforce/environments/gridworld"
require_relative "reinforce/environments/taxi"
require_relative "reinforce/environments/taxiv2"
require_relative "reinforce/algorithms/sarsa"
require_relative "reinforce/algorithms/temporal_difference"

module Reinforce
  
  # Calculate the moving average of the rewards.
  def self.moving_average(data, window_size)
    data.each_cons(window_size).map { |window| window.sum / window.size.to_f }
  end
  
  class Error < StandardError; end
end

# Torch-dependent components are optional at load time.
begin
  require "torch"
  require_relative "reinforce/q_function_ann"
  require_relative "reinforce/algorithms/dqn"
  require_relative "reinforce/algorithms/reinforce"
  require_relative "reinforce/algorithms/ppo"
rescue LoadError => error
  raise unless error.path == "torch"
end
