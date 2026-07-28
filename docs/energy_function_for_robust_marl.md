# Applying the Energy-Function Method to Belief-Based Robust Cooperative MARL

> **Purpose.** This document sets up a research question: can the *energy-function method for
> Bayesian Nash Equilibria* (summarized separately, and assumed to be in your context) be
> adapted to train the policies in a *belief-based robust cooperative multi-agent RL* game?
> It describes the target game, why the current training method fails, and the specific
> obstacles and design questions involved in porting the energy function over. Read the
> energy-function summary FIRST; this document assumes familiarity with it (KKT residuals,
> $\Psi_i$, dual-player networks, Monte-Carlo estimation, single-objective energy minimization).

---

## 1. What I'm working on (the target game)

**Robust cooperative multi-agent reinforcement learning (robust c-MARL) against Byzantine
adversaries.** A team of $N$ agents shares a single reward and acts over a sequential task
(a Markov / partially-observable Markov game, e.g. a repeated matrix game or StarCraft-like
environment). At the start of each episode, nature secretly assigns each agent a hidden
**type** $\theta_i$:

- $\theta_i = 0$: cooperative (wants to maximize the shared team reward)
- $\theta_i = 1$: adversarial / compromised (wants to *minimize* the team reward, i.e. worst-case)

The type is **private and fixed for the episode**. Cooperative agents do not know who is
compromised; they must **infer it from observed behavior** (a belief) and act accordingly.
Currently the setting assumes **at most one adversary** per episode; my planned extension is
**k independent (non-colluding) adversaries**, $1 < k < N$.

This is a **Bayesian game** in the same sense as the energy-function paper — private types,
strategies conditioned on type/belief — but with three important differences from the
continuous-type BNE setting that method was built for (see Section 4).

### The equilibrium I'm solving for
An **ex interim robust Markov perfect Bayesian equilibrium**: each cooperative agent
best-responds to its posterior belief over others' types, against the worst-case adversary;
the adversary best-responds to the team; beliefs are Bayes-consistent. It is a **saddle point**
of a max–min problem: the team maximizes the shared return, the adversary (controlling the
compromised agent) minimizes it. This is essentially a **team-vs-adversary** structure
(cooperative within the team, zero-sum between team and adversary) — a special case of a
general-sum game, closely related to "adversarial team Markov games."

---

## 2. The networks currently being trained

Four neural networks (all trainable; two are discarded at deployment):

1. **Team policy** $\pi$ (shared across cooperative agents) — input: agent's observation +
   its belief; output: action distribution. **The deliverable.**
2. **Adversary policy** $\hat\pi$ — the worst-case attacker controlling the compromised agent.
   A *training tool* (sparring partner), discarded at deployment.
3. **Critic** $V$ — value estimate, centralized (sees global state), training only. Discarded
   at deployment. Used to compute advantages for policy gradients.
4. **Belief network** — a GRU (recurrent) outputting $P(\theta_i \mid \text{history})$ for each
   agent. Trained by **cross-entropy against the true type** (available during centralized
   training). Deployed alongside the policy. For the k-adversary extension this is a
   **factored belief**: $N$ independent per-agent corruption probabilities, avoiding an
   exponential distribution over adversary subsets.

The team policy and adversary form the **max–min pair**; the critic and belief are auxiliary.

---

## 3. Why the current training method fails (the motivation)

Training currently uses **two-timescale gradient descent–ascent**: the team ascends the
expected return, the adversary descends it, on separated learning rates (adversary faster).
This is the standard approach (used by the two papers this work builds on, EIR-MAPPO and
BATPAL), optionally with PPO clipping.

**Observed problem: limit cycles.** Empirically, the reward-under-attack does NOT converge to
a stable value — it **oscillates violently** (in one run swinging between ~0.00 and ~0.99
indefinitely). The team and adversary keep leapfrogging each other. This is the classic
pathology of gradient descent–ascent on saddle-point / min–max problems: because you are
chasing a saddle rather than minimizing a scalar, **nothing is monotonically decreasing**, so
there is no convergence guarantee and the dynamics fall into a cycle around the equilibrium
instead of landing on it. Where you stop determines what number you report — the result is
seed-dependent and unreliable.

**This is exactly the disease the energy-function method could cure.** The energy function
converts a coupled saddle/fixed-point problem into a **single non-negative scalar minimization**
whose zero is the equilibrium. A minimization cannot cycle the way a saddle chase does — you can
only descend. If we could define a valid energy function for this robust-MARL game, we would
replace the unstable min–max dynamics with a stable descent toward energy zero, and we would
gain a **clean convergence diagnostic** (watch the energy $\to 0$) that the current setup
completely lacks.

---

## 4. What I want the Fable model to help me figure out

**Core question:** How do we adapt the energy-function method (KKT-residual energies + dual
networks + single-objective minimization) to train the team policy and adversary in this robust
c-MARL game?

The natural first formulation, by direct analogy to the energy paper: define the game's
**stationarity residuals** for both players and minimize

$$
\text{Energy} \;=\; \big\|\nabla_{\theta_{\text{team}}} J\big\|^2 \;+\; \big\|\nabla_{\theta_{\text{adv}}} J\big\|^2
$$

(optionally plus complementarity terms for action-space constraints), where $J$ is the
(estimated) expected team return. Energy $= 0$ iff both players are simultaneously stationary,
i.e. at a robust equilibrium. Then do plain gradient descent on this single energy — one
unified minimization, no saddle chase, no leapfrogging.

### The obstacles that make this a research problem (not a trivial port)

The energy-function paper was built for a setting that differs from mine in three ways. The
Fable model should reason carefully about each:

1. **Expected return vs. closed-form utility.**
   In the BNE paper, $\rho_i$ (interim utility) and $\nabla_{a_i} u_i$ are available in
   closed form; Monte Carlo is only over opponent *types*. In RL, the "utility" is an
   **expected discounted return over trajectories** — there is no closed-form $\nabla_a u$.
   The gradient becomes a **policy-gradient estimate** (sampled, high-variance). The naive
   energy $\|\nabla J\|^2$ then estimates $\mathbb{E}[\hat g^2]$, which is a **biased** proxy
   for $\|\mathbb{E}[\hat g]\|^2$ — it includes the gradient's *variance*, not just its mean.
   **Key question:** how to define/estimate the energy so that sampling noise does not make
   "energy $=0$" unreachable (the variance floor problem)? (E.g. separate mini-batches for the
   two factors of the square; control variates; or an energy function $\psi$ chosen — as the
   paper notes is allowed — to be something *other* than the naive square, with better
   noise properties.)

2. **Sequential / dynamic structure (Markov game, not one-shot).**
   The BNE paper is a one-shot game: a strategy maps a type to an action. My game is
   **sequential** — a policy maps a *history* to an action over many timesteps, with the
   Bellman/temporal-credit structure. The "KKT/stationarity conditions" of a Markov game
   involve the *value function* and the policy-gradient theorem, not a static $\nabla_a u$.
   **Key question:** what are the correct stationarity residuals for a robust Markov game, and
   can the energy be built on the policy-gradient / Bellman stationarity conditions (possibly
   using the existing critic $V$ as the machinery that supplies $\nabla J$)?

3. **Discrete actions + non-convexity.**
   The BNE paper assumes continuous actions, strongly-concave $\rho_i$ in own action (so KKT is
   necessary AND sufficient), and smooth activations. My policies are typically **discrete-action,
   nonconvex–nonconcave**. So (a) constraints are simplex constraints on action *probabilities*
   rather than box constraints on continuous actions — the dual-network / complementarity
   machinery may map to the policy simplex; and (b) $\|\nabla J\|^2 = 0$ holds at *every*
   stationary point, including spurious local ones, not only the global robust equilibrium.
   **Key question:** does energy minimization still help (converge to *a* stationary point,
   stably) even though it cannot guarantee the *global* robust equilibrium? (Note: this is no
   worse than gradient descent–ascent, which also only finds local solutions AND cycles — so
   "stably reaches a good stationary point" would already be a win.)

### What "success" would look like
- The energy $\to 0$ (or a small floor) monotonically-ish, giving a **convergence certificate**.
- Reward-under-attack **stabilizes** (low variance across late iterations and across seeds)
  instead of cycling.
- Ideally competitive with or better than two-timescale GDA / PPO on the same environments
  (toy repeated matrix game first; then LBF / SMAC).

### Why this is novel / worth doing
- The two reference papers (EIR-MAPPO, BATPAL) train with two-timescale GDA (+PPO) and both
  exhibit the instability. **Energy-function training for robust MARL equilibria has not been
  done.**
- There is adjacent min–max literature (optimistic/extragradient methods with last-iterate
  convergence; "Hamiltonian gradient descent" which minimizes $\|\nabla\|^2$ of the game vector
  field — very close in spirit). But my group's specific energy-function machinery — including
  the freedom to define $\psi$ as something *other* than the naive square, and the dual-player
  network treatment of constraints — may offer advantages (noise-robustness, constraint
  handling on the simplex) that generic Hamiltonian descent does not. Differentiating from that
  literature is part of the contribution.

---

## 5. Concrete asks for the Fable model

1. **Formulate the energy function** for this robust Markov game: what are the correct
   stationarity (and complementarity) residuals for the team policy and the adversary policy,
   given that "utility" is an expected return and actions are discrete? Show how the existing
   critic supplies the needed gradients.
2. **Address the noise/bias problem** in $\|\nabla J\|^2$ under sampled policy gradients —
   propose an estimator or an alternative $\psi$ that keeps "energy $= 0$" attainable.
3. **Handle the simplex constraints** on the action distributions via the dual-player /
   complementarity mechanism, analogous to the paper's box-constraint treatment.
4. **Relate to / differentiate from** Hamiltonian gradient descent and optimistic/extragradient
   methods.
5. **Sketch the training algorithm** (what to compute per step, how it replaces the
   two-timescale GDA loop) and how to keep it tractable (the paper's backward pass is quadratic
   in the number of players due to strategic coupling — with $N$ agents + adversaries this
   matters; note the team shares one policy network via parameter sharing, which may help).

---

## 6. Minimal glossary bridging the two settings

| Energy-function paper | This robust MARL problem |
|---|---|
| player $i$ with private type $\theta_i$ | agent $i$ with hidden cooperative/adversarial type |
| strategy $f_i(\theta_i)$ (type → action) | policy $\pi(\cdot\mid \text{obs}, \text{belief})$ (history → action) |
| interim utility $\rho_i$ (closed form) | expected discounted return $J$ (sampled, no closed form) |
| $\nabla_{a_i}\rho_i$ (closed form) | policy-gradient estimate (via critic/advantages) |
| KKT stationarity residual | policy-gradient stationarity / Bellman optimality residual |
| box constraints on $a_i$ | simplex constraints on action probabilities |
| dual-player network $\lambda_i$ | (analogous) multiplier network for simplex constraints |
| energy $\Psi_i \ge 0$, $=0$ at KKT | game energy $\ge 0$, $=0$ at robust equilibrium |
| single-objective minimization = BNE | single-objective minimization = robust MPBE (goal) |
| one-shot game | sequential Markov game |
| continuous actions, strongly concave | discrete actions, nonconvex–nonconcave |
| Monte Carlo over opponent types | Monte Carlo over trajectories AND types |
