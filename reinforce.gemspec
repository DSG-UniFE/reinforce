# frozen_string_literal: true

require_relative "lib/reinforce/version"

Gem::Specification.new do |spec|
	spec.name = "reinforce"
	spec.version = Reinforce::VERSION
	
	spec.summary = "Reinforcement Learning suite for Ruby"
	spec.authors = ["Mauro Tortonesi"]
	spec.license = "MIT"
	
	spec.homepage = "https://github.com/mtortonesi/reinforce"
	
	spec.metadata = {
		"documentation_uri" => "https://mtortonesi.github.io/",
		"homepage_uri" => "https://github.com/mtortonesi/reinforce",
		"source_code_uri" => "https://github.com/mtortonesi/reinforce",
	}
	
	spec.files = Dir['{examples,lib}/**/*', '*.md', base: __dir__]
	
	spec.required_ruby_version = ">= 3.0"
	
	spec.add_dependency "torch-rb", "~> 0.14.1"
end
