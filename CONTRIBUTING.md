# Contributing to reinforce

This document covers the practical mechanics of working on this codebase: setup, running tests, linting, and coverage. For the pull request process, the Developer Certificate of Origin, and the code of conduct, see the "Contributing" section of `readme.md` and `code_of_conduct.md` -- this document doesn't duplicate those.

## Tech stack

This project deliberately favors a few less-mainstream Ruby gems over the usual defaults (see `AGENTS.md` for the short version):

- [`sus`](https://github.com/socketry/sus), not RSpec, for testing.
- [`bake`](https://github.com/ioquatix/bake), not Rake, for task running.
- [`standard`](https://github.com/standardrb/standard) for linting/formatting.
- [`covered`](https://github.com/socketry/covered), not SimpleCov, for coverage.
- `gems.rb`, not `Gemfile` (Bundler supports both; this project uses the former). `gems.locked` is intentionally **not** committed: this is a gem, not an application, so your own `bundle install` resolves a lockfile against `reinforce.gemspec`'s version constraints rather than reusing a pinned snapshot.

## Setup

This project depends on [torch.rb](https://github.com/ankane/torch.rb), which needs a matching LibTorch install to compile its native extension. `.github/workflows/ci.yml` is the source of truth for the exact version and commands; as of this writing:

```bash
# macOS (Apple Silicon)
curl -L https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.13.0.zip -o libtorch.zip
unzip -q libtorch.zip
bundle config set build.torch-rb --with-torch-dir="$(pwd)/libtorch"

# Linux (CPU)
curl -L https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.13.0%2Bcpu.zip -o libtorch.zip
unzip -q libtorch.zip
bundle config set build.torch-rb --with-torch-dir="$(pwd)/libtorch"
```

For other platforms, GPU builds, or a different LibTorch version, see torch.rb's own readme, including its LibTorch/torch.rb compatibility table -- torch.rb pins to a specific LibTorch minor version, and building against a mismatched one fails at compile time with confusing errors.

Then install dependencies as usual:

```bash
bundle install
```

## Running tests

```bash
bundle exec sus
```

To run a subset, pass file paths. One gotcha: most test files assume `lib/reinforce.rb` has already loaded (e.g. so `Reinforce::Agent`/`Reinforce::Environment` exist), which only happens when the whole suite loads together. Running a single file directly, e.g. `bundle exec sus test/reinforce/algorithms/sarsa.rb`, fails with an unrelated `NameError: uninitialized constant Reinforce::Agent` -- that's this load-order gap, not a real problem with the file. If you need to focus on one area, it's usually faster to run the full suite and read the relevant part of the output.

## Learning benchmark

```bash
bundle exec sus benchmark/learning.rb
```

The tests above check that the code runs; this checks that the algorithms still *learn*. Each case trains an algorithm from scratch under several fixed seeds on GridWorld or Taxi, runs the greedy policy, and requires a minimum number of seeds to reach the goal. It takes about two and a half minutes, so it is kept out of the default `sus` run; CI runs it as a separate job. When a case fails, it prints a per-seed report (success, return, steps, training time). See the comment at the top of `benchmark/learning.rb` for how the budgets and thresholds were chosen before changing them.

### Exploration benchmark

```bash
bundle exec sus benchmark/exploration.rb
```

Measures exploration instead of learning: the fraction of reachable states an agent visits (via `Reinforce::Wrappers::StateCoverage`) within a fixed budget on a sparse-reward 21x21 four-rooms GridWorld. Today it only checks a random-policy baseline, to keep the testbed hard enough to tell methods apart; every exploration method (count-based bonuses, RND, ...) should add a case that beats it. The header of `benchmark/exploration.rb` records the current numbers for DQN, PPO and SAC.

### Skill discovery benchmark

```bash
bundle exec sus benchmark/skills.rb
```

Checks that unsupervised skill discovery (`DIAYN`) learns skills that are genuinely different with no reward at all: each of 8 skills is run on an open 10x10 GridWorld, and the discriminator must attribute most runs' final states to the right skill (untrained skills sit at chance level, which a second case checks). The file's header records the numbers and DIAYN's known limitation on four-rooms layouts.

## Linting

```bash
bundle exec standardrb          # check
bundle exec standardrb --fix    # auto-fix what's safe to auto-fix
```

CI runs this as a blocking check across the whole repository with zero tolerated offenses -- there's no grandfathering file list for old style debt, so if you touch a file that has any, please fix it rather than working around it.

## Coverage

```bash
COVERAGE=BriefSummary bundle exec sus     # overall percentage + least-covered files
COVERAGE=PartialSummary bundle exec sus   # only the uncovered code snippets
bundle exec bake covered:validate --minimum 0.90   # what CI actually checks
```

CI fails if aggregate line coverage across `lib/**/*.rb` drops below 90%. That threshold is a floor guarding against regressions, not a target to write towards -- `PPO` and `ExDM` are currently this project's biggest known coverage gaps, and closing them is more valuable than defending the number.

## Code conventions

- `lib/<path>.rb` and `test/<path>.rb` mirror each other 1:1 (e.g. `lib/reinforce/algorithms/sarsa.rb` corresponds to `test/reinforce/algorithms/sarsa.rb`). Keep new files consistent with this.
- A class's file name matches its class name (`SARSA` in `sarsa.rb`, `MonteCarloPolicyGradient` in `monte_carlo_policy_gradient.rb`, and so on) -- including when the "obvious" name would collide with a `Reinforce::*` module. A class defined directly under `Reinforce::Algorithms` whose bare name matches a top-level `Reinforce::*` constant (e.g. a class named `Reinforce`, or `Agent`) shadows that constant for any unqualified reference written elsewhere in the `Reinforce::Algorithms` namespace, due to how Ruby's `Module.nesting`-based constant lookup works. See the comments atop `lib/reinforce/algorithms/monte_carlo_policy_gradient.rb` and `lib/reinforce/algorithms/ppo.rb` for the specific instances this bit previously.
- Online algorithms -- ones that interact with an environment episode-by-episode (`SARSA`, `DQN`, `PPO`, `SAC`, `MonteCarloPolicyGradient`) -- share one `train(episodes:, steps_per_episode:, **kwargs)` signature. Offline algorithms -- ones that train from a fixed, pre-collected dataset (`BehaviorCloning`, `ConservativeQLearning`, `ExDM`) -- share `train(epochs:, batch_size:)` instead. These are genuinely different shapes of training loop; don't force a new algorithm into the wrong one just to match its neighbors.
- Every algorithm's hyperparameters live in a nested `Config` (a `Data.define` whose `initialize` supplies the defaults), resolved with `Reinforce::Hyperparameters.resolve(Config, config, hyperparameters)` and exposed as `#config`. Constructors take `(collaborators..., config = nil, **hyperparameters)`: environments, datasets, models and optimizers are collaborators, not hyperparameters. Add the new class to the table in `test/reinforce/hyperparameters.rb`, which checks this contract for every algorithm. Don't hard-code a tunable value inside a method: make it a `Config` field, so tests and users can change it without `instance_variable_set`.
- New environments should `include Reinforce::Environment`; new algorithms should `include Reinforce::Agent`. Both raise a clear `NotImplementedError` naming the missing method if you forget to implement part of the contract, instead of a `NoMethodError` raised from deep inside whichever caller happens to hit it first.
