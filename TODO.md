# TODO

Planned changes to `reinforce`, roughly in the order we intend to tackle them. Items are removed from this list once they land; see `changelog.md` for what has already shipped.

## Consolidation of the existing algorithms

- **Experiment tracker integration.** An MLflow sink for `Reinforce::Metrics` (via MLflow's REST API, so no new gem dependency), and a way to record each run's hyperparameters (`agent.config.to_h`) alongside its metrics.
- **Device support everywhere.** Pass `device:` through the tensor construction call sites in SAC, PPO, `BehaviorCloning`, `ConservativeQLearning`, `ExDM` and `MonteCarloPolicyGradient`, following the pattern already used in `QFunctionANN` and `DQN`.
- **Close the remaining coverage gaps,** starting with `ppo.rb` and the environments (`taxi.rb`, `game_2048.rb`, `1d_walk.rb`), plus `experience.rb`.

## Reward-free / unsupervised RL

- **Exploitation with bonuses in DQN and SAC.** PPO's separate intrinsic value head solved sparse four rooms (5/5 seeds); DQN and SAC still add the bonus to the reward and have no equivalent. Twin Q-heads for the intrinsic reward (or a decaying bonus) would be the analogue.
- **SAC and exploration bonuses from coordinate observations.** From raw (row, column) coordinates, a count bonus doesn't change SAC's coverage (0.43-0.56), while from one-hot observations it raises it from 0.27-0.35 to 0.86-0.93; probably its critic smooths the bonus across neighboring cells. Lower priority now that one-hot exists.
- **Move ExDM's `ScoreBonus` onto the intrinsic-reward interface,** now that RND (its well-understood relative) is there.
- **Show where RND beats counts:** a variant of the testbed where exact counts break down (e.g. noisy or continuous observations, so no state is ever seen twice).
- **Intrinsic Curiosity Module (ICM)**, built on the same intrinsic-reward module.
- **DIAYN skills that go somewhere.** On four-rooms layouts DIAYN's skills stay near the start. An exploration bonus alone doesn't change that, nor do room-only discriminator features alone; both together reach 2-4 of 4 rooms (mean 2.8 over 5 seeds). Next candidates: longer training, or a distance-aware objective (LSD, Park et al. 2022) that rewards skills for covering ground.
- **Use learned skills downstream:** a high-level controller that picks DIAYN skills to solve a task (the hierarchical-RL item below), or fine-tuning one skill on a task's reward.
- **Successor features**, so a policy can be adapted to new reward weightings without retraining. Needs a task family to evaluate on (e.g. GridWorld with several goal positions).
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
