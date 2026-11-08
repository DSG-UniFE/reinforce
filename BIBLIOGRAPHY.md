## Important papers

The following papers played a relevant role in reinforce's implementation.

### Foundations

(Sutton & Barto, 2018)
Sutton and Barto, "Reinforcement Learning: An Introduction", 2nd edition, MIT Press, 2018, http://incompleteideas.net/book/the-book-2nd.html.
Used for: `TemporalDifference`, n-step `SARSA`, and the naming of `MonteCarloPolicyGradient`.

(Watkins & Dayan, 1992)
Watkins and Dayan, "Q-learning", Machine Learning 8, 1992.
Used for: `TemporalDifference` (off-policy mode).

(Rummery & Niranjan, 1994)
Rummery and Niranjan, "On-line Q-learning using connectionist systems", Technical Report CUED/F-INFENG/TR 166, Cambridge University, 1994.
Used for: `TemporalDifference` (on-policy mode) and `SARSA`.

### Value-based deep RL

(Mnih et al., 2015)
Mnih et al., "Human-level control through deep reinforcement learning", Nature 518, 2015.
Used for: `DQN` and its target network.

(Schaul et al., 2015)
Schaul et al., "Prioritized Experience Replay", 2015, https://arxiv.org/abs/1511.05952.
Used for: `PrioritizedExperienceReplay` and `SumTree`.

(Van Hasselt et al., 2015)
Van Hasselt, Guez and Silver, "Deep Reinforcement Learning with Double Q-learning", 2015, https://arxiv.org/abs/1509.06461.
Used for: `DQN`'s `double_dqn:` option.

(Wang et al., 2016)
Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning", 2016, https://arxiv.org/abs/1511.06581.
Used for: `Models::DuelingQNetwork` and `DQN`'s `dueling:` option.

### Policy gradient and actor-critic

(Williams, 1992)
Williams, "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning", Machine Learning 8, 1992.
Used for: `MonteCarloPolicyGradient` (REINFORCE).

(Schulman et al., 2017)
Schulman et al., "Proximal Policy Optimization Algorithms", 2017, https://arxiv.org/abs/1707.06347.
Used for: `PPO`.

(Huang et al., 2022)
Huang et al., "The 37 Implementation Details of Proximal Policy Optimization", ICLR Blog Track, 2022, https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/.
Used for: `PPO`'s cleanrl-style defaults (orthogonal initialization, Tanh activations, gradient clipping).

(Haarnoja et al., 2018a)
Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor", 2018, https://arxiv.org/abs/1801.01290.
Used for: `SAC`.

(Haarnoja et al., 2018b)
Haarnoja et al., "Soft Actor-Critic Algorithms and Applications", 2018, https://arxiv.org/abs/1812.05905.
Used for: `SAC`'s automatic entropy tuning.

(Christodoulou, 2019)
Christodoulou, "Soft Actor-Critic for Discrete Action Settings", 2019, https://arxiv.org/abs/1910.07207.
Used for: `SAC`'s discrete-action formulation.

### Offline RL and off-policy evaluation

(Kumar et al., 2020)
Kumar et al., "Conservative Q-Learning for Offline Reinforcement Learning", 2020, https://arxiv.org/abs/2006.04779.
Used for: `ConservativeQLearning`.

(Le et al., 2019)
Le, Voloshin and Yue, "Batch Policy Learning under Constraints", 2019, https://arxiv.org/abs/1903.08738.
Used for: `Offline::Evaluation::TabularFQE` (fitted Q evaluation).

(Precup et al., 2000)
Precup, Sutton and Singh, "Eligibility Traces for Off-Policy Policy Evaluation", ICML, 2000.
Used for: `Offline::Evaluation.weighted_importance_sampling`.

### Network building blocks

(He et al., 2016)
He et al., "Deep Residual Learning for Image Recognition", 2016, https://arxiv.org/abs/1512.03385.
Used for: `Networks::ResidualBlock`.

(Vincent et al., 2008)
Vincent et al., "Extracting and Composing Robust Features with Denoising Autoencoders", ICML, 2008.
Used for: `Models::DenoisingAutoencoder` (`ExDM`'s intrinsic bonus).
