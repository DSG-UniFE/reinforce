# frozen_string_literal: true

# Released under the MIT License.
# Copyright, 2023, by Mauro Tortonesi.

source 'https://rubygems.org'

# Specify your gem's dependencies in reinforce.gemspec
gemspec

# gem 'rake', '~> 13.0'
gem 'bake', '~> 0.24.1'
gem 'bake-gem', '~> 0.12.1'
gem 'bake-modernize', '~> 0.50.0'
gem 'standard', '~> 1.15.0'

group :test do
  gem 'covered', '~> 0.28.1'
  gem 'sus', '~> 0.35.2'
end
