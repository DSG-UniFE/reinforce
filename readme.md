# Reinforce

This is a Reinforcement Learning (RL) library built on top of Ruby at the University of Ferrara, Italy. 
The library is in its early stages of development and is not yet ready for production use. 

At the moment, it is simply a playground that we set up to learn some technical and/or implementation details of RL algorithms. We hope that in time it could grow and become a mature product.

Reinforce requires the torch.rb gem, which provides Ruby bindings for the PyTorch library. 

## Prerequisites

The installation of torch.rb might require some tweaks depending on your system. 
On MacOs you can install libtorch using Homebrew:
    
    $ brew install pytorch

On Linux instead you need to first download libtorch from [here](https://pytorch.org/), and then extract the archive in a folder of your choice.
Then you need add the following configuration to your project:


    $ bundle config build.torch-rb --with-torch-dir=/path/to/libtorch

This will configure the bundler to build torch.rb using the libtorch installation in the specified directory.

For more info, please visit the torch.rb's [github repo](https://github.com/ankane/torch.rb) 

## Installation

Install the gem and add to the application's Gemfile (or gems.rb) by executing:

    $ bundle add reinforce

If bundler is not being used to manage dependencies, install the gem by executing:

    $ gem install reinforce


## Usage

Train a DQN agent to solve the GridWorld environment:
    
    $ bundle exec examples/dqn_gridworld.rb

By default the DQN policy is saved. You can test the trained policy by executing:

    $ bundle exec examples/dqn_gridworld_test.rb

## Define a new environment

Defining a new environment is fairly simple. Use the examples environment as guide in defining your own.
All you need is to wrap your environment in a class that defines the following methods:
1. `initialize` - Initialize the environment.
2. `reset` - Reset the environment to its initial state.
3. `state_size` - Return the size of the state space.
4. `actions` - Return the action space; you can retrieve the number of actions by calling `actions.size`.
5. `step` - Execute an action in the environment and return the next state, reward, done, and info.
6. `render` - Render the environment on specified output, e.g, `$stdout` (optional).

## Contributing

Many thanks to Prof. Mauro Tortonesi and Filippo Poltronieri who are currently developing the library.
All contributions are welcome.

We welcome contributions to this project.

1.  Fork it.
2.  Create your feature branch (`git checkout -b my-new-feature`).
3.  Commit your changes (`git commit -am 'Add some feature'`).
4.  Push to the branch (`git push origin my-new-feature`).
5.  Create new Pull Request.

## License

This software is available as open source under the terms of the [MIT License](https://opensource.org/licenses/MIT).

### Developer Certificate of Origin

This project uses the [Developer Certificate of Origin](https://developercertificate.org/). All contributors to this project must agree to this document to have their contributions accepted.

### Contributor Covenant

This project is governed by [Contributor Covenant](https://www.contributor-covenant.org/). All contributors and participants agree to abide by its terms.


