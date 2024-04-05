# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi.

source 'https://rubygems.org'

# Specify your gem's dependencies in reinforce.gemspec
gemspec

# gem 'rake', '~> 13.0'
gem 'bake', '~> 0.18.2'
gem 'bake-gem', '~> 0.4.0'
gem 'bake-modernize', '~> 0.17.8'
gem 'unicode_plot'
gem 'torch-rb', git: 'https://github.com/ankane/torch.rb.git'
gem 'standard', '~> 1.3'

group :test do
  gem 'covered', '~> 0.21.0'
  gem 'sus', '~> 0.21.1'
end
