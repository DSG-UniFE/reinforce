# Reinforce

This is a Reinforcement Learning (RL) library built on top of Ruby at the [Big
Data and Compute Continuum Research Lab](https://bdcc.unife.it) of the
University of Ferrara, Italy. reinforce is still in its early stages of
development and is not yet ready for production use.

At the moment, it is simply a playground that we set up to learn some technical
and/or implementation details of RL algorithms. We hope that in time it could
grow and become a mature product.


## Prerequisites

Reinforce requires the [torch.rb](https://rubygems.org/gems/torch-rb) gem,
which provides Ruby bindings for the libtorch library (the C++ core that
underpins the PyTorch framework).

The installation of torch.rb might require some tweaks depending on your
system. First of all, you'll need to install a development version of libtorch,
the C++ core of Pytorch.

On MacOs you can just install libtorch (and Pytorch) using Homebrew:

    $ brew install pytorch

On Linux instead you need to install the package that your distribution uses to
ship a development version of libtorch (python-pytorch on Arch, libtorch-dev -
or, even better, pytorch - on Ubuntu, etc.).

If your Linux distribution doesn't package libtorch, you'll need to download it
from [the Pytorch Web site](https://pytorch.org/), and then extract the archive
in a folder of your choice. Then you need add the following configuration to
your project:

    $ bundle config build.torch-rb --with-torch-dir=/path/to/libtorch

This will configure the bundler to build torch.rb using the libtorch
installation in the specified directory.

For more info, please visit [torch.rb's github repo](https://github.com/ankane/torch.rb).

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

## Shields

A shield corrects an agent's actions before they are executed and steers its
training toward safe actions. `Reinforce::Shields` reasons about safety with
[Logic Tensor Networks](https://github.com/mtortonesi/LTNtorch-rb), so it needs
the optional `ltntorch` gem and its own `require`:

```ruby
require "reinforce/shields"

environment = Reinforce::Environments::MultiCluster.hard(clusters: 4)
safety = Reinforce::Shields::MultiClusterSafety.new(environment)
shield = Reinforce::Shields::AllocationShield.new(safety, preference: :latency)
agent = Reinforce::Algorithms::ContinuousSAC.new(environment, shield: shield)
agent.train(episodes: 60, steps_per_episode: 500)
```

`MultiClusterSafety` scores an allocation of service replicas with the
probability that it is safe: probabilistic sensors estimate whether each
cluster has the resources its replicas need despite delayed telemetry, and an
LTN formula combines them. `AllocationShield` keeps allocations scoring at least
a threshold, searches for safer ones by gradient ascent otherwise, and gives the
agent a differentiable safety loss. `examples/shielded_sac_multi_cluster.rb`
compares SAC with and without the shield.

## Graph networks

`Reinforce::Graph` holds building blocks for graph neural networks, for
environments whose state is a graph (such as a Computing Continuum of
locations, links and services):

- `Graph::MessagePassing`: one round of message passing (an edge model, sum
  or mean aggregation, a node model), on one node set or between two
  (bipartite graphs), optionally returning updated edges;
- `Graph::Temporal`: a recurrent (GRU) state per node across steps;
- `Graph.aggregate`, `Graph.bidirectional` and `Graph.batch` (many graphs
  as one, for fast training).

Graphs are plain tensors: node features, an edge index `[2, edges]` and
optional edge features.

## Define a new environment

Defining a new environment is fairly simple. Use the examples environment as guide in defining your own.
All you need is to wrap your environment in a class that defines the following methods:
1. `initialize` - Initialize the environment.
2. `reset` - Reset the environment to its initial state.
3. `state_size` - Return the size of the state space.
4. `actions` - Return the action space; you can retrieve the number of actions by calling `actions.size`.
   Environments with continuous actions define `action_space` instead, returning a `Reinforce::Spaces::Box`;
   environments that make several discrete choices per step (PPO) return a `Reinforce::Spaces::MultiDiscrete`
   and take an Array of indices in `step`.
5. `step` - Execute an action in the environment and return the next state, reward, done, and info.
6. `action_mask` - Which actions are allowed in the current state (optional): an Array of booleans, one per
   action, or for a `MultiDiscrete` space one such Array per dimension. PPO and DQN never choose a masked
   action, and DQN bootstraps only over the actions allowed next.
7. `render` - Render the environment on specified output, e.g, `$stdout` (optional).

## Contributing

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


