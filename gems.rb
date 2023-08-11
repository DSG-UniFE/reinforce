# frozen_string_literal: true

source 'https://rubygems.org'

# Specify your gem's dependencies in reinforce.gemspec
gemspec

# gem 'rake', '~> 13.0'
gem 'bake', '~> 0.18.2'
gem 'bake-gem', '~> 0.4.0'
gem 'bake-modernize', '~> 0.17.8'

gem 'standard', '~> 1.3'

group :test do
  gem 'covered', '~> 0.21.0'
  gem 'sus', '~> 0.21.1'
end
