# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi.

require_relative "reinforce/version"

module Reinforce
  
  # Cal+culate the moving average of the rewards
  def self.moving_average(data, window_size)
    data.each_cons(window_size).map { |window| window.sum / window.size.to_f }
  end
  
  class Error < StandardError; end
  # Your code goes here...
end
