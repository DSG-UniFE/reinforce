# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi.

require_relative "reinforce/version"
puts "#{__dir__}"
# Require environments, algorithms, and helper functions
Dir["#{__dir__}/reinforce/*.rb"].each { |file| require_relative ".#{file.gsub(__dir__, '')}" }
# require_relative all files in reinforce/algorithms
Dir["#{__dir__}/reinforce/algorithms/*.rb"].each { |file| require_relative ".#{file.gsub(__dir__, '')}" }
# require_relative all files in reinforce/environments
Dir["#{__dir__}/reinforce/environments/*.rb"].each { |file| require_relative ".#{file.gsub(__dir__, '')}" }

module Reinforce
  
  # Cal+culate the moving average of the rewards
  def self.moving_average(data, window_size)
    data.each_cons(window_size).map { |window| window.sum / window.size.to_f }
  end
  
  class Error < StandardError; end
  # Your code goes here...
end
