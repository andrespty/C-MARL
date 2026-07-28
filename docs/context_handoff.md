# Research Context Handoff: Belief-Based Robust Cooperative MARL

This document captures the full context of an ongoing research project so a new assistant can continue seamlessly.

---

## 1. The research area

**Robust cooperative multi-agent reinforcement learning (robust c-MARL)** against **Byzantine failures** — settings where a team of agents sharing a common reward must stay effective when one (or more) teammate is compromised and acts adversarially (arbitrary, worst-case behavior). The identity of the compromised agent is unknown and must be **inferred from behavior** during execution.

Framed as a **Bayesian game**: nature secretly assigns each agent a hidden *type* (cooperative vs. adversarial). Agents maintain a **belief** over teammates' types, update it from observed behavior (Bayes' rule / learned approximation), and best-respond. Solution concept: an **ex interim robust Markov perfect Bayesian equilibrium** (each agent conditions on its posterior belief), which provably **weakly dominates** the older **ex ante** approach (fixed prior, treat everyone as possibly adversarial) as t→∞, because beliefs converge to the truth.

---

## 2. The two central papers

### Paper 1 — Li et al. 2024 (ICLR), "Byzantine Robust Cooperative MARL as a Bayesian Game" (EIR-MAPPO)
- **Model:** BARDec-POMDP. Binary types θ_i ∈ {0,1} (0=cooperative, 1=worst-case adversary). **Exactly one adversary** per episode (or none). Type fixed at episode start.
- **Prior:** p(no adversary)=0.5; p(agent i is adversary)=0.5/N.
- **Solution:** ex interim robust Markov perfect Bayesian equilibrium. Proves existence (Kakutani), weak domination over ex ante (Prop 2.3), convergence of a robust Harsanyi-Bellman equation (Banach contraction).
- **Algorithm (EIR-MAPPO):** built on MAPPO. Four networks:
  - policy π (shared across cooperative agents; conditions on obs + belief)
  - adversary π̂ (worst-case attacker)
  - critic V (centralized, training only)
  - belief network (GRU; outputs P(θ|history))
  - **Two-timescale**: adversary updates fast, team slow. Belief trained by cross-entropy against the true type (known during centralized training). Policy gradient masked by θ so each timestep's data trains the right network.
- **Attack model:** perturbation P^α replaces the compromised agent's action with the adversary's; cooperative agents' actions pass through unchanged. The adversary's action is always drawn from π̂, but π̂ can choose to act benignly (attack timing is learned, not a separate switch).
- **Perturbation formula:** `P^α(ā|a,π̂,θ) = ∏_i [π̂^i(·)·θ^i + δ(ā^i−a^i)·(1−θ^i)]` — Dirac delta = "leave cooperative agent's action unchanged."
- **Results:** matches MAPPO cooperative performance; far more robust under attack (near oracle "True Type"). Toy game, LBF, SMAC.

### Paper 2 — Kazari & Dán 2026 (ICLR), "Bayesian Robust Cooperative MARL Against Unknown Adversaries" (BATPAL)
- **Model:** Bayesian Dec-POMDP with a **continuum of adversarial types** θ_i ∈ [0,1]. Still **at most one adversary** per episode (‖θ‖₀ ≤ 1). The [0,1] interval encodes *severity/objective* of that single adversary, NOT a count of adversaries.
- **Key idea:** critiques EIR-MAPPO's worst-case-only assumption. Real adversaries have diverse objectives; training against only worst-case overfits. Instead, **partition adversarial policies by severity** η = (Vmax − V(π0,ρ))/(Vmax − Vmin), discretize into K bins, train one policy robust against a representative (worst-case-in-bin) adversary from each bin.
- **Externally-constrained RL (EC-PPO):** finds each severity-level adversary via constrained optimization (objective and constraint live on two *different* MDPs). Log-barrier + PPO clipping.
- **Belief:** RNN, cross-entropy against true type; captures severity but (ablation shows) NOT the exact attack type.
- **Objective:** Bayesian regret. Solution: perfect Bayesian equilibrium of the discretized game.
- **Results:** BATPAL generalizes to unseen/non-worst-case attacks better than EIR-MAPPO. Environments: LBF, MPE-Spread, SMAC-2s3z, SMAC-MMM.
- **Multi-victim:** only an appendix sketch (D.5) via a "compound agent" hack; flags combinatorial blowup as unsolved.

---

## 3. The user's proposed research direction

**Extend from 1 adversary to k adversaries (1 < k < N), INDEPENDENT (non-colluding).** User settled on:
- **Independent adversaries** (not colluding) — keeps it a clean multi-label extension, avoids team-vs-team.
- Likely generative story: each agent independently corrupted w.p. q (Binomial count), possibly capped.
- **k is a knob to sweep**, not fixed. Don't commit to N/2; sweep k/N and find where robustness degrades empirically.

### Why this is a real gap (confirmed by literature sweep)
- **Ex ante, multi-adversary** is DONE: RADAR (Phan et al. 2021) handles a subset of adversaries without belief.
- **Ex interim (belief-based), multi-adversary** is the OPEN cell. EIR-MAPPO and BATPAL are both single-adversary.
- A related paper to READ: "Defending Against Unknown Corrupted Agents" (OpenReview 62pcoIqovw) proposes **ARNEQ** (Adversarially Robust Nash Equilibrium) for n agents / k corrupted, and warns the induced game is NOT zero-sum and NOT an adversarial team Markov game — confirming existence proofs get harder with k>1. User should read this to check overlap before claiming novelty.

### The core technical challenges of the k-extension
1. **Belief blowup:** belief over *subsets* is exponential (~2^N). **Fix: factored belief** — N independent per-agent sigmoids `b_i = P(agent i corrupted)`. Linear in N, but loses correlation structure (can't express "exactly one of {2,3}" or the ≤k cap coupling). Whether this matters is a key ablation.
2. **Equilibrium/existence proof:** no longer team-vs-one; needs rework (loses the clean ATMG/Kakutani structure).
3. **Belief collapse near k≈N/2:** inference relies on spotting deviators against a cooperative majority. As k→N/2 the majority reference dissolves → belief should degrade. This is the interesting *phenomenon* to demonstrate, connected to the classical **Byzantine fault-tolerance N/3 threshold** (Lamport et al. 1982). Sweep k/N, look for an RL-analog tolerance threshold.

### The Bellman equation changes for k adversaries
- min over multiple adversaries (joint, but independent)
- belief sum ranges over subsets ‖θ‖₀ ≤ k → **exponential blowup**, fixed via factored product approximation
- perturbation replaces up to k actions instead of 1

### Belief-sum blowup mitigation options (discussed)
- **Factored belief** (mean-field): N marginals — but the *sum* over θ is still 2^N terms unless combined with:
- **Monte Carlo**: sample a few θ from the factored belief, average.
- **Amortization**: feed the belief vector directly into critic/policy; let the network learn the belief-averaged value (closest to what the papers actually do).
- Recommended combo: factored belief + amortized networks (+ MC if an explicit expectation is ever needed).

### General-sum discussion (a bigger, separate direction — NOT the current focus)
- Direction A (arbitrary-objective adversaries, team intact): incremental; BATPAL's severity-collapse is the bottleneck.
- Direction B (fully general-sum, every agent self-interested): a whole research program; breaks the shared-value equilibrium, needs full Bayesian Markov perfect equilibrium, richer belief (infer objective direction, not just severity), new robustness objective (Bayesian regret / coalitional).

---

## 4. Code status (the working prototype)

Language: **TensorFlow / Keras**, functional style, in Jupyter. Refactored into files (Option A — pass an `Agents` object explicitly, no globals):
- `game_environments/ToyGame.py` — vectorized toy env
- `networks.py` — build_policy, build_adversary, build_critic, build_belief
- `misc.py` — sample_types_batched, belief_features_batched, sample_batched, random_rollout_batched
- `rollout.py` — make_belief_step, run_episodes_batched
- `updates.py` — compute_returns/advantages_batched, update_{critic,belief,policy,adversary}
- `agents.py` — Agents dataclass + build_agents
- `train.py` — train loop

### The toy environment (from Han et al. 2022, used in Paper 1)
- N=2 agents, 2 states, binary actions, 100 timesteps.
- s1 = XNOR (reward if agents agree), s2 = XOR (reward if agents differ). Transition: agree→s1, disagree→s2.
- Fully observable (each agent sees the state).
- **Currently N=2, k=1** to match the paper before extending.

### Four networks
- policy, adversary, critic = **MLPs** (memoryless — act/value on current input).
- belief = **GRU** (needs memory — integrates history to infer types). This is the architecturally special one and where the k-extension difficulty lives.
- All networks are **batch-agnostic** — batching required NO network changes, only env/rollout/update changes.

### Key implementation details / fixes made
- **Egocentric belief fix (CRITICAL):** originally the shared policy got the same input for both agents → couldn't specialize defense → reward-under-attack collapsed. Fix: call the policy per-agent with belief **rolled** so each agent's own slot is index 0 (`np.roll(b_t, -i, axis=1)`), so "my partner's corruption probability" is in a consistent slot. This unstuck the defense.
- **Belief training:** cross-entropy against TRUE type θ (available during centralized training). NOT computed via Bayes' rule — the network learns the posterior for free because cross-entropy's minimizer IS the posterior.
- **Batching:** run B episodes in parallel (vectorized env `ToyEnvVec`/`ToyGame`). MLP updates flatten (B,T,...) → (B*T,...); belief keeps (B,T,...) for the GRU. Batched agents into single network calls (B*N rows) instead of per-agent Python loops.
- **Speed optimizations:** biggest costs were `update_belief` (627ms, full GRU over (B,T,feat)) and the rollout (447ms, sequential 100-step loop with per-step `.numpy()` syncs). Wrapping the training steps in `@tf.function` (inner pure-TF step + outer numpy-prep wrapper, with explicit dtype casts to avoid retracing) helped meaningfully. `@tf.function` on the tiny MLP *inference* calls was negligible. Levers: lower B, update belief less often, subsample/truncate belief sequences.

### Training dynamics observed (IMPORTANT)
- Belief network works great (→99% accuracy).
- Clean reward (no adversary) → ~0.99 (perfect coordination).
- **Reward under attack OSCILLATES violently** — the classic min-max limit cycle. One run ended stably at 0.995 (success); another cycled and ended mid-collapse at 0.578 with belief acc crashing to 0.53. **The convergence is a limit cycle, not a reliable fixed point.** Where you stop determines what you see. Result is seed-dependent.
- This is the known instability of naive min-max policy gradient — the papers' PPO clipping + careful two-timescale + PopArt are designed to tame it. The prototype uses raw policy gradients, so it reproduces both the mechanism AND the instability.

### Success criterion (agreed)
NOT reward(adv)=1.0. A worst-case adversary controlling 1 of 2 agents *should* hurt the team. Success = reward under attack settles **clearly above** the ~0.25–0.5 an undefended team gets, ideally approaching clean reward. Getting stable ~0.5–0.9 would validate the method.

### Current hyperparameters
- hidden=128, lr_policy=5e-4, lr_adv=1e-3 (adversary faster), lr_critic=5e-4, lr_belief=5e-4
- gamma=0.99, ep_len=100, gamma normalization on advantages (across B*T)
- B experimented at 256–1000; team_interval and belief update frequency are tuning knobs.

---

## 5. Immediate next steps (where the conversation left off)

1. **Stabilize the limit cycle.** The user explicitly does NOT want PPO clipping (both papers already use it) — they want a NOVEL method not yet applied to this problem. See the companion file `novel_directions.md`.
2. Run **multiple seeds** to characterize the variance (the single-run results are unreliable).
3. Then extend to **k independent adversaries** with the factored belief, and run the **k/N sweep** looking for a Byzantine-style tolerance threshold.
4. Read the ARNEQ paper (OpenReview 62pcoIqovw) and RADAR (Phan et al. 2021) to precisely position novelty.

---

## 6. Key conceptual clarifications established (so the user doesn't re-litigate them)
- "reward(adv)" = team reward WHEN an adversary is present → want it HIGH (not 0). Separate from "adv" the advantage variable, and from the adversary network's objective (which wants team reward low).
- Type is fixed per episode (not switching mid-episode). Belief converges to a fixed hidden truth — that's why Bayesian consistency arguments work.
- Ad hoc teamwork = cooperative twin (infer unknown-but-friendly teammate); robust c-MARL = infer whether teammate is secretly adversarial. Same Harsanyi-Bellman machinery; difference is friend-vs-disguised-enemy, which forces min-max equilibrium.
- Byzantine = worst-case arbitrary failure (from Lamport et al. 1982 distributed systems; classical N/3 tolerance threshold; Yin et al. 2018 ported it to ML).
