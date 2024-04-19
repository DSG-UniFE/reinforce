  # Create a class that performs the same operation of a DummyVectorizedEnvironment
module Reinforce
  class DummyVectorizedEnvironment
    def initialize(environment, num_envs)
      @environment = environment
      @num_envs = num_envs
      @states = Array.new(num_envs) { @environment.reset }
    end

    def state_size
      #warn "environment.state_size: #{@environment.state_size}"
      @environment.state_size
    end

    def step(actions)
      #warn "Applying actions to vectorized environment---> actions: #{actions}"
      actions = [actions] if actions.is_a?(Integer)
      nobs = []
      nrews = []
      ndones = []
      @states.zip(actions).map do |state, action|
        #warn "state: #{state}, action: #{action}"
        nob, nrew, ndone = @environment.step(action) 
        nobs << nob
        nrews << nrew
        ndones << ndone
      end
      [nobs, nrews, ndones]
    end 

    def reset
      @states = Array.new(@num_envs) { @environment.reset }
    end

    def action_masks
      @environment.action_masks
    end

    def actions
      @environment.actions
    end

    def render(output)
      @environment.render(output)
    end
  end
end