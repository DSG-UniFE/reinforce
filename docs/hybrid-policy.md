# The hybrid shielded policy

This document describes how reinforce combines a neural policy with a
Logic Tensor Network (LTN) shield for continuous allocation actions, as
used by `Algorithms::ContinuousSAC` (SAC-LTN) and `Algorithms::PPO`
(PPO-LTN) with `Shields::AllocationShield` and
`Shields::MultiClusterSafety`. It also places the construction among the
alternatives and lists its known weaknesses.

## 1. The construction in one paragraph

A neural actor proposes an allocation (the *base* action). An LTN scores
the probability that it is safe, from probabilistic sensors (see
[sensor-model.md](sensor-model.md)). At run time, a base action scoring
below a threshold is corrected by a short gradient search through the LTN.
At training time, the same LTN provides a safety loss that pushes the
actor's own proposals toward safe allocations. The reinforcement learning
algorithm learns from the base actions, with the rewards of the corrected
(executed) ones. It is therefore a **hybrid**: train-time alignment
through a logic loss, plus run-time refinement by the logic.

```
observation ──► actor ──► base action a ──► shares q⁰ ──► LTN: Sa(o, q⁰) ≥ τ ? ──yes──► execute q⁰
                  ▲                                            │ no
                  │                                            ▼
                  │                              gradient search through the LTN
                  │                                            │
                  │                                            ▼
                  │                                   execute q⁺ (cheapest safe, or safest)
                  │                                            │
   RL update on (o, a, reward of q⁺) ◄─────────────────────────┘
   + λ_s × safety loss (through the LTN, into the actor)
```

## 2. Components

### Actions and shares

An action a ∈ [−1, 1]^(N+1) (N clusters and a reject option) becomes
allocation shares by Euclidean projection onto the probability simplex:
q⁰ = Π_Δ((a + 1) / 2) (`Shields::Simplex`). Executed shares q are sent
back to the environment as 2q − 1.

### The safety formula

The knowledge is one formula (`MultiClusterSafety#evaluate`), over a
variable s (an observation and candidate shares) and a variable c (the
clusters):

```
safe(s)       := (∃c: chosen(s,c) ∧ full_fit(s,c) ∧ ¬tight(s,c))
                 ∨ (reject_chosen(s) ∧ reject_safe(s))
full_fit(s,c) := cpu_fit(s,c) ∧ mem_fit(s,c) ∧ gpu_fit(s,c) ∧ bw_fit(s,c)
tight(s,c)    := tight_penalty(s) ∧ ¬comfortable(s,c)
```

`chosen(s, c)` is grounded in the actor's share for cluster c, and the
other predicates in the sensors. The operators are the product t-norm
(x ∧ y = xy), the standard negation (1 − x), Łukasiewicz's disjunction
(min(1, x + y)) and the bounded-sum existential (min(1, Σ)). With them,
the truth value of `safe` is exactly the probability that a replica sent
to a destination drawn from the shares lands safely, under the
corresponding probabilistic logic program:

    Sa(o, q) = Σ_c q_c f_c (1 − ρ (1 − h_c)) + q_rej j(q).

The product is exact because the sensor events are taken as independent,
and the bounded sum because the destinations of a replica are mutually
exclusive (`test/reinforce/shields/multi_cluster_safety.rb` checks the
equality). With other knowledge, the operators must again match the
dependencies between the events, or the score becomes a fuzzy
approximation.

### Run-time refinement

`AllocationShield#call`: if Sa(o, q⁰) ≥ τ, q⁰ is executed. Otherwise K
gradient steps (step size η) on unconstrained values v, starting at q⁰
and projected onto the simplex, maximize

    Sa(o, q) − λ_d ‖q − q⁰‖²       while Sa(o, q) < τ,
    −(‖q − q⁰‖² + β ℓ(o, q))        once Sa(o, q) ≥ τ,

where ℓ is the mean latency to the request's dependency. Among q⁰ and the
K candidates, the shield executes the cheapest that reaches τ, or, if none
does, the safest. Defaults: τ = 0.6, K = 6, η = 0.5, λ_d = 0.05, β = 0.5.

### What the learner sees

The learner is trained on the **base** action: SAC stores it in the
replay buffer, PPO keeps it (as the pre-tanh sample) and its
log-probability in the rollout, in both cases with the reward and next
observation of the executed action. The shield is a fixed, deterministic
function of the observation and the base action, so the environment
together with the shield is a stationary Markov decision process, and both
algorithms optimize the policy for that process: the value of a base
action is the value of proposing it under the shield.

### The safety loss

Added to the actor's loss with weight λ_s (`safety_weight`, default 0.1):

- **deterministic** (SAC-LTN, and PPO-LTN with `safety_samples: 0`):
  L_s = E_o[−log max(Sa(o, q_θ(o)), ε)], on the actor's deterministic
  action;
- **Monte Carlo** (PPO-LTN with `safety_samples: M`, default 8):
  L_s = E_o[−log max((1/M) Σ_m Sa(o, q_θ^(m)(o)), ε)], over M
  reparameterized samples of the policy, tanh(μ + σ ξ_m). It estimates
  −log P(safe) of an action drawn from the policy, so its gradient reaches
  the spread σ as well as the mean.

The gradient flows through the sensors and the LTN into the actor; the
run-time search is detached from it. In PPO, a target KL divergence
(`target_kl`) stops a rollout's updates early, because the ratio clipping
does not limit how far the safety loss moves the policy.

## 3. Where it sits among the alternatives

Neurosymbolic reinforcement learning combines logic with a policy in three
broad ways:

| | Train-time alignment | Run-time refinement or masking | Hybrid (ours) |
|---|---|---|---|
| **How** | the actor is trained with a logic satisfiability loss | the actor proposes; the logic masks, re-weights or corrects the action | both |
| **Guarantees at run time** | soft: rules are penalties | strong: unsafe actions are not executed | strong |
| **Foresight** | the policy learns long-term consequences | the refiner fixes one step greedily | both |
| **Changing a rule** | needs retraining | immediate | immediate, the policy catches up |
| **Cost per decision** | one forward pass | the logic, over candidates | the logic, over candidates |

Within the hybrid family, our choices are:

- **Refinement by gradient search**, not masking or re-weighting. Masking
  fits discrete actions (it is what `Shields::DiscreteShield` does for PPO
  and DQN: options below the threshold are masked, and the joint action is
  checked again). For continuous shares a search keeps the correction close
  to the proposal.
- **Learning from base actions**, not executed ones (section 4).
- **One formula with probabilistic semantics**, rather than several fuzzy
  axioms combined by implication, so that the score is a probability the
  threshold τ can be read against.
- **Analytic groundings** (the sensors), not learnt predicates; learnt
  groundings with the rules held fixed are a planned extension.

## 4. Known weaknesses

- **Shield dependence.** Learning from base actions with the rewards of
  corrected ones means an unsafe proposal that the shield fixes earns the
  reward of the fix. The policy then has no reward-driven reason to stop
  proposing it, and only the safety loss pushes it away. On the monitored
  testbed (4 clusters, 100k training steps), the shield still intervened on
  about one test step in ten after training, for PPO-LTN and SAC-LTN alike;
  with less calibrated sensors, interventions were far more frequent.
  Storing executed actions instead
  is possible for SAC (an off-policy learner) but not directly for PPO,
  whose ratio needs the log-probability of the executed action under the
  policy, which a deterministic correction does not have.
- **Greedy refinement.** The search fixes one step at a time; long-term
  effects (fragmentation, saturating a cluster ahead of a peak) are left to
  the policy.
- **Cost.** The K-step search, run for every decision, dominates the time
  per step (several milliseconds, against a fraction of a millisecond for
  the policy alone).
- **The score is as good as the sensors.** Calibration (see
  [sensor-model.md](sensor-model.md)) is the check.

## 5. Variants worth testing

- **Closed-form re-weighting of the shares**,
  q′_c ∝ q_c ψ_c^β with ψ_c the probability that cluster c accommodates its
  quota (and a corresponding rule for the reject share), with β ≥ 1 as a
  strictness parameter. No search, so much cheaper, at the cost of not
  aiming at a threshold.
- **A differentiable refiner during training**: unroll the K search steps
  in the actor's loss, so that the actor learns what the shield will do to
  its proposals.
- **Executed-action learning for SAC**, against base-action learning.
- **An intervention cost** added to the training reward, as a direct
  incentive against relying on the shield.

## Code

| Piece | Where |
|---|---|
| Simplex projection | `lib/reinforce/shields/simplex.rb` |
| Bounded-sum and product quantifiers | `lib/reinforce/shields/bounded_sum.rb`, `product.rb` |
| Sensors and safety formula | `lib/reinforce/shields/multi_cluster_safety.rb` |
| Run-time refinement and safety loss | `lib/reinforce/shields/allocation_shield.rb` |
| Discrete masking shield | `lib/reinforce/shields/discrete_shield.rb` |
| Sensor calibration | `lib/reinforce/shields/calibration.rb` |
| SAC-LTN | `lib/reinforce/algorithms/continuous_sac.rb` (`shield:`) |
| PPO-LTN | `lib/reinforce/algorithms/ppo.rb` (`shield:`, `safety_samples:`, `target_kl:`) |
| Benchmark | `examples/ppo_ltn_multi_cluster.rb` |
