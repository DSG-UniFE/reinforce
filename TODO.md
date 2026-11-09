# TODO

Planned changes to `reinforce`, roughly in the order we intend to tackle them. Items are removed from this list once they land; see `changelog.md` for what has already shipped.

## Consolidation of the existing algorithms

- **Shared metrics recording.** Replace the per-algorithm `@logs` hashes, which use different key sets, with a common `Reinforce::MetricsRecorder` that all algorithms push to. Use the same names for common metrics (`episode_reward`, `episode_length`, `loss`) and allow algorithm-specific extras.
- **Learning regression benchmark.** Train PPO, DQN and SAC on GridWorld and Taxi for a fixed episode budget across several seeds, and check a minimum reward or success-rate threshold. Tag it as a slow test so the default `sus` run stays fast, and run it in CI as a separate step.
- **Hyperparameter config objects.** Replace the long positional argument lists in each algorithm's constructor with per-algorithm `Data.define` configs (e.g. `DQNConfig`), so a full hyperparameter set is one constructible, loggable, diffable value.
- **Device support everywhere.** Pass `device:` through the tensor construction call sites in SAC, PPO, `BehaviorCloning`, `ConservativeQLearning`, `ExDM` and `MonteCarloPolicyGradient`, following the pattern already used in `QFunctionANN` and `DQN`.
- **Close the remaining coverage gaps,** starting with `ppo.rb` and the environments (`taxi.rb`, `game_2048.rb`, `1d_walk.rb`), plus `experience.rb`.

## Reward-free / unsupervised RL

- **Random Network Distillation (RND)** as an opt-in intrinsic-reward module that any algorithm can use.
- **Intrinsic Curiosity Module (ICM)**, built on the same intrinsic-reward module.
- **Count-based exploration** as a baseline for the small discrete environments.
- **DIAYN** for reward-free skill discovery.
- **Successor features**, so a policy can be adapted to new reward weightings without retraining.
- *Stretch:* Forward-Backward representations for zero-shot RL.

## Continuous control and further algorithms

- **Action-space abstraction** (`Discrete`, `Box`, `MultiDiscrete`), with environments declaring their space explicitly instead of algorithms inferring it from `actions.size`.
- **Gaussian policy heads** for SAC and PPO, and **TD3** as a deterministic continuous-control baseline.
- **Distributional value learning** (C51 or QR-DQN).
- **Constrained / safe RL**, starting with Lagrangian PPO.
- **Model-based RL**, starting with a PETS- or MBPO-style probabilistic ensemble dynamics model.
- **Hierarchical RL / options**, as a home for skills discovered by DIAYN.
- **Decision Transformer** for offline RL from logged traces.
- A convolutional building block, once an environment with image or grid-shaped observations exists.

## Multi-agent RL

- **Multi-agent environment interface** (parallel API with per-agent observations, actions, rewards and dones, plus an optional global state for centralized training).
- **VDN** and **QMIX** for cooperative discrete tasks.
- **MAPPO**, extending the existing PPO with a centralized critic.
- **MADDPG** for general-sum settings with continuous actions.
- **COMA** for counterfactual credit assignment.
- **Parameter sharing** across homogeneous agents as a first-class option; mean-field MARL for large agent counts.
- **Computing Continuum environment:** a task-offloading/placement simulator with heterogeneous edge, fog and cloud nodes, network latency, energy cost and SLA constraints. It will be built as a single-agent environment first, then in a multi-agent version.

## Cross-cutting

- **Benchmarking harness:** fixed seeds, a standard environment set, and recorded learning curves (mean ± spread across seeds).
- **Documentation:** an algorithm selection guide and generated YARD API docs, once the interfaces above stabilize.
- **1.0 release** once the consolidation items above are done.
