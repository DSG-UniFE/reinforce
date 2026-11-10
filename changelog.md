# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Breaking

- Every algorithm (and `QFunctionANN`) now takes its hyperparameters as a
  `Config` value object -- e.g. `Reinforce::Algorithms::PPO::Config` -- and/or
  as keyword overrides on top of it, and exposes the result as `#config`
  (`agent.config.to_h` gives a loggable hash). See `Reinforce::Hyperparameters`.
  Positional hyperparameters are gone:

  | Before | After |
  |---|---|
  | `PPO.new(env, lr, policy, value, clip, epochs, mb, gamma, ...)` | `PPO.new(env, learning_rate: lr, policy_model: policy, value_model: value, ...)` |
  | `DQN.new(env, lr, gamma, epsilon, ...)` | `DQN.new(env, learning_rate: lr, discount_factor: gamma, epsilon:, ...)` |
  | `SARSA.new(env, q_function, epsilon)` | `SARSA.new(env, q_function, epsilon:)` |
  | `QFunctionANN.new(states, actions, lr, gamma, ...)` | `QFunctionANN.new(states, actions, learning_rate: lr, discount_factor: gamma, ...)` |
  | `network_config:` (every algorithm) | `network:` |

  `QFunctionANN`'s and `PPO`'s learning rate, previously required, now
  default to 1e-3 and 2.5e-4. A misspelled hyperparameter raises
  `ArgumentError` instead of being silently ignored.
- Hyperparameters that used to be hard-coded are now configurable:
  `DQN`'s `training_start` (1000), `q_update_frequency` (10),
  `target_update_frequency` (500), `tau` (1.0) and `minibatch_size` (128),
  and `PPO`'s `gae_lambda` (0.97). `PPO`'s `max_grad_norm` now defaults to
  0.2 on its own rather than to `clip_param`.

### Added

- `GridWorld` takes `observation: :one_hot` (a `size * size` vector with a 1.0
  at the agent's cell) besides the default `:coordinates`; positions
  (`#state`, `#obstacles`, `#reachable_states`) are unchanged. With one-hot
  observations a count-based bonus raises the four-rooms coverage of all
  three of PPO (0.95-0.97), DQN (0.75-0.80) and SAC (0.86-0.93).
- Intrinsic rewards (exploration bonuses): `Reinforce::IntrinsicRewards`
  defines a small interface (`#observe`, `#call`, `#train`), and
  `IntrinsicRewards::CountBased` implements the count-based bonus
  coefficient / sqrt(N(s)). `PPO`, `DQN` and `SAC` take one as
  `intrinsic_reward:`; PPO adds it while collecting rollouts, DQN and SAC
  when sampling from replay (so the bonus reflects current counts), and all
  record it as an `intrinsic_reward` metric while `episode_reward` stays
  extrinsic. On the four-rooms exploration benchmark it raises PPO's state
  coverage from 0.37-0.55 to 0.86-0.93 (DQN: 0.62-0.83 to 0.68-0.79; SAC is
  unaffected so far).
- Exploration testbed: `GridWorld` takes `reward: :sparse` (+1 only at the
  goal) besides the default `:dense`, `GridWorld.four_rooms` builds the
  classic four-rooms layout (sparse by default), and `GridWorld#reachable_states`
  lists the cells reachable from the start. New environment wrappers:
  `Reinforce::Wrappers::RewardFree` (zero reward, original in
  `info[:extrinsic_reward]`) for reward-free pretraining, and
  `Reinforce::Wrappers::StateCoverage`, which measures exploration (distinct
  states visited, and coverage of the reachable states) for any agent.
  A new `benchmark/exploration.rb` runs in CI.
- `Reinforce::Metrics`: every algorithm now records its training metrics into
  a `Metrics::Recorder`, exposed as `#metrics` (`#logs` remains as an alias
  and reads the same way as before, e.g. `agent.metrics[:loss]`). Online
  algorithms all record `episode_reward`, `episode_length` and `loss`. Output
  goes to pluggable sinks: `Metrics::ConsoleSink` prints periodic one-line
  summaries, `Metrics::JSONLinesSink` writes every value to a JSON-lines file,
  and any `call(key, value, step)` object (e.g. a lambda) works as a sink.
- Experimental offline-RL building blocks: `OfflineDataset`, fitted Q evaluation,
  behaviour cloning, and conservative Q-learning.
- A discrete-action Soft Actor-Critic (SAC) implementation with twin critics,
  target critics, replay sampling, optional automatic entropy tuning, and
  model checkpoint save/load support.
- An experimental Exploratory Diffusion Model (ExDM) prototype for offline data,
  including score-based intrinsic-reward components.
- Tests for offline datasets, offline evaluation, SAC, ExDM, and the offline-RL
  integration pipeline.
- Deterministic Torch test helpers, including `REINFORCE_TEST_SEED` support for
  reproducing randomized test failures.
- A macOS GitHub Actions workflow that runs the test suite and verifies that
  the built gem can be loaded.

### Changed

- `SAC` now normalizes rewards by default too (`normalize_rewards: true`),
  adapted to its replay buffer: the buffer keeps raw rewards, and each
  sampled batch is divided by the *current* running standard deviation of
  the discounted return when computing critic targets. This took SAC from
  22/25 to 25/25 seeds solved across GridWorld 5x5, 10x10 and Taxi. ExDM,
  which feeds SAC transitions directly rather than through `#train`, is
  unaffected.
- `PPO` now normalizes rewards by default, like Gymnasium's
  `NormalizeReward`: each reward is divided by a running estimate of the
  standard deviation of the discounted return, so the critic's targets stay
  near unit scale whatever the environment's reward scale. Pass
  `normalize_rewards: false` for the previous behavior. On GridWorld this
  took PPO from 7/10 to 10/10 seeds solved (greedy policy, 5x5 and 10x10).
  The running statistics are kept across `#train` calls but not saved by
  `#save`. Adds `Reinforce::RunningMeanStd`.
- `SAC`: new defaults `warmup_steps: 2_000` (was 100) and, with automatic
  entropy tuning, a target entropy of `0.3 * log(|actions|)` (was 0.98, which
  kept the policy nearly uniform). Together with the random warmup above,
  SAC went from 1/3 to 9/10 seeds solved on fixed-layout Taxi and from 0/5
  to 8/10 on a 10x10 GridWorld (greedy policy). The target-entropy default
  also applies to `ExDM`, which builds on SAC.
- `PPO`: the learning rate now actually anneals linearly to zero over
  training, as intended (Integer division made it constant). PPO with a low
  learning rate may need more rollouts than before; `examples/ppo_taxi.rb`'s
  settings still learn.
- `PPO`: gradient-norm clipping now has its own `max_grad_norm:` option
  instead of reusing `clip_param` (the default, 0.2, is unchanged).

### Fixed

- `SARSA`: `episode_reward` metrics are per episode; they used to accumulate
  across all episodes of a `#train` call.
- `SAC`: the first `warmup_steps` environment steps now take uniformly
  random actions (as in the reference SAC implementations) instead of
  sampling the randomly initialized policy, and warmup is counted across
  `#train` calls instead of restarting on each one.
- `PPO`: an episode truncated by the end of a rollout now bootstraps from
  the state it reached, not from the environment's reset state.
- `PPO`: `episode_reward` metrics are per episode; they used to accumulate
  across all episodes within a rollout.
- `PPO`: Integer observations returned by `#step` no longer crash training,
  and `#train` no longer mutates the observation arrays returned by the
  environment's `#reset`.

- `DQN`, `PPO` and `SARSA` no longer print training progress by default; pass
  `progress_output: $stdout` to `#train` to get the old `Training: N%` output.
- `Taxi#step` no longer prints "Passenger picked up!"/"Task Completed!" to
  stderr; it reports them in the step info as `{event: :passenger_picked_up}`
  and `{event: :task_completed}` instead.
- Load Torch-dependent algorithms only when `torch-rb` is available, while
  preserving support for loading the non-Torch core of the library.
- Improved PPO model initialization and training-mode handling.

### Notes

- SAC currently supports discrete action spaces only.
- ExDM is experimental and is not yet a full implementation of the published
  continuous-control, unsupervised-RL algorithm. It should not be used for
  benchmark claims until its implementation and evaluation are completed.

## [0.2.0] - 2024-04-02

### Added

- Beta release.

## [0.1.0] - 2023-07-03

### Added

- Initial release.
