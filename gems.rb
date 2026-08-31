# frozen_string_literal: true

source 'https://rubygems.org'

# Specify your gem's dependencies in reinforce.gemspec
gemspec


group :examples do
  gem 'unicode_plot', '>= 0.0.5'
end

group :test do
  gem 'covered', '>= 0.30.0'
  gem 'sus', '>= 0.37.2'
end

group :development do
  gem 'bake', '>= 0.25.0'
  gem 'bake-gem', '>= 0.14.0'
  gem 'bake-modernize', '>= 0.57.1'
  gem 'ostruct', '>= 0.6.3' # no longer included in Ruby, so we need to explicitly add it as a dependency; required by standard
  gem 'tsort', '>= 0.2.0' # no longer included in Ruby, so we need to explicitly add it as a dependency; required by standard
  gem 'standard', '>= 1.56'
end
