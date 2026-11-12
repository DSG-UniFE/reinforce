# frozen_string_literal: true

# Shields: safety layers that correct an agent's actions before they are
# executed and guide its training toward safe actions (see
# Reinforce::Algorithms::ContinuousSAC's `shield:`).
#
# They reason with Logic Tensor Networks, so they need the optional ltntorch
# gem and are not loaded by `require "reinforce"`:
#
#   require "reinforce/shields"
#   safety = Reinforce::Shields::MultiClusterSafety.new(environment)
#   shield = Reinforce::Shields::AllocationShield.new(safety, preference: :latency)
#   agent = Reinforce::Algorithms::ContinuousSAC.new(environment, shield: shield)
require_relative "../reinforce"
require_relative "shields/simplex"
require_relative "shields/bounded_sum"
require_relative "shields/allocation_shield"
require_relative "shields/multi_cluster_safety"
