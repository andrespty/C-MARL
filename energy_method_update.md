# Energy Method for Belief-Based Robust c-MARL: Progress Update

## Problem
A team of cooperative agents must stay effective when a teammate is secretly compromised
(Byzantine) and acts against the team. Agents keep a belief over who is compromised and
act on it (ex interim robust equilibrium, as in EIR-MAPPO, Li et al. 2024).

The standard training method, two-timescale gradient descent-ascent (GDA) with PPO, does
not converge in its last iterate: the team and adversary cycle around the equilibrium.
Co-trained metrics hide this; a freshly trained adversary exploits the final team.

## The method
Find the robust Bayesian equilibrium by **minimizing one energy** that is zero exactly at
equilibrium, instead of running gradient descent-ascent. Per iteration:

1. **Critic:** learn Q_i(o, theta, t, a) with a TD target, using the true types.
2. **Energy:** for every step and agent, the soft optimality residual

   Psi = sum_a ( tau * log p(a) - (sigma * Qbar(a) - V_tau) )^2,
   V_tau = tau * logsumexp(sigma * Qbar / tau)

   - Qbar is **belief-averaged** (ex interim): sum over type profiles weighted by the belief.
   - The critic is **detached** inside the energy (it cannot lower the energy itself).
   - sigma = +1 for the team, -1 for the adversary.
   - Team and adversary **both descend** E (one minimization, no ascent), plus a decaying
     team policy-gradient **selection term** to pick the high-reward equilibrium
     (uniform play also has zero energy).
3. **Belief:** GRU trained with cross-entropy against the true types.

This is the KKT system of the entropy-regularized per-state problem. Its joint zero is a
quantal response equilibrium (QRE), which tends to a Nash equilibrium as tau -> 0.

### What made it work
- **log_softmax** for log-probabilities (log of clamped probabilities had a dead gradient).
- **epsilon-exploration** in rollouts ((1 - eps) * pi + eps * uniform), so every action
  has data. Without it, Q for unplayed actions is a guess and the energy can be zero at a
  fake equilibrium (the certificate is fooled).
- **Time input** to the critic (finite-horizon returns depend on steps left).

## Results so far (Toy game, single seed)
Robust = per-step team reward against a freshly trained PPO adversary (max 0.5).
Score = 0.5 * clean + 0.5 * robust (ideal 0.75, random play 0.50).

| Method                                   | Clean | Robust | Score | Notes                                   |
|------------------------------------------|-------|--------|-------|-----------------------------------------|
| GDA (official-code faithful)             | 0.99  | 0.24   | 0.62  | exploitable at every checkpoint         |
| Energy, Fischer-Burmeister (KKT)         | 0.99  | 0.08   | 0.54  | energy ~0 but fake: unplayed-action Q   |
| Energy, soft, tau = 1                    | 0.50  | 0.50   | 0.50  | stuck at uniform equilibrium            |
| Energy, KL                               | 1.00  | 0.00   | 0.50  | collapsed to deterministic play         |
| **Energy, soft + log_softmax + eps**     | 0.99  | **0.49** | **0.74** | energy -> 0.004 while robustness rises |
| EIR-MAPPO (paper, Fig. 6 diagonal)       |   -   | 0.44   |   -   | 5 seeds x 2 attacks                     |

Key observation: in the final version the energy falls **while** true robustness rises,
and both settle together, with no checkpoint selection (last iterate). The measured
exploitability (0.011 per step) is within the soft bound tau * log|A| ~ 0.14.

## Improvements

**Two fixes not yet applied**
1. **Soft TD target.** Train the critic on the soft Q:
   y_t = r_t + gamma * sum_a' p(a') * ( Q(s_{t+1}, a') - tau * log p(a') ).
   Then tau * log p = Q - V_tau is an exact QRE fixed point instead of a target that
   moves whenever the policy changes.
2. **Q normalization** (running mean/std). Makes tau mean the same thing in every
   environment; needed for generalization.

**Also worth adding**
3. **tau annealing** (e.g. 1.0 -> 0.1): recovers clean reward and moves from a QRE toward
   a Nash equilibrium; the certificate bound tau * log|A| shrinks with it.
4. **KL energy with epsilon-exploration on:** its value is directly the soft
   exploitability in reward units, the cleanest certificate.
5. **Scaling to k adversaries** (main contribution): factored belief, a few sampled type
   profiles instead of summing over all of them in Qbar, and a belief for adversaries.
6. **Better selection term:** lexicographic (maximize team value subject to E <= eps)
   instead of a hand-decayed coefficient.
7. **Report critic error next to the energy:** the certificate is only as good as Q.

## Next steps (evaluation)
- Paper protocol: 5 seeds x 2 attacked agents, energy vs GDA, mean +/- 95% CI in episode
  reward (paper EIR-MAPPO: 44.14 out of 50).
- Ablations: eps = 0; FB vs soft vs KL (with eps); no belief averaging.
- A second game with a different equilibrium structure (e.g. deterministic robust
  response, 3+ actions) before tuning further on the toy game.
