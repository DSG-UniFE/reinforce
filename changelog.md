# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
