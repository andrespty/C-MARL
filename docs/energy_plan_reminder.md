# My Project + The Energy-Function Plan (plain-language reminder)

A note to myself. Explains what I'm building, what's broken, and the plan to fix it.
Written simply so I can re-read it later and remember why each piece matters.

---

## 1. The project (brief)

**What:** A team of N agents shares a reward. One (or more) is secretly *corrupted* and
sabotages the team. Nobody knows who. Each agent must **infer who's corrupted from behavior**
and **defend** accordingly.

**Networks I train:**
- **Team policy** — picks cooperative actions (the thing I actually keep)
- **Adversary** — the saboteur (a training sparring partner; thrown away after)
- **Belief** — guesses who's corrupted (kept)
- **Critic** — estimates how good a situation is (training helper; thrown away)

**Goal:** train these so the team stays strong even when attacked. The team *maximizes* reward,
the adversary *minimizes* it — a **max-min game**.

**The core problem I'm trying to solve:** training this max-min game with the standard method
(two-timescale gradient descent-ascent) **doesn't converge — it oscillates forever** (a "limit
cycle"). The team and adversary keep leapfrogging. My reward-under-attack swings wildly and
never settles, so results are unreliable and seed-dependent.

**The idea:** use my group's **energy function** method. Instead of chasing a max-min saddle
(which cycles), turn the whole thing into **minimizing one number** that is zero exactly at the
solution. You can't cycle when you're just going downhill. Bonus: watching the energy hit zero
gives me a **convergence certificate** — proof I've actually arrived.

---

## 2. The points (each: what it is → the problem → the fix)

### Point 0 — The warning: my naive idea already has a name

**What:** My first instinct — energy = `‖gradient of team‖² + ‖gradient of adversary‖²` — is
already a known method called **Hamiltonian Gradient Descent**.

**Problem:** If I just do that, it's not novel. Reviewers will say "this already exists."

**Fix:** The novelty has to come from *how I make it different*. There are three real ways
(Points 1–3 below). If I do those, it stops being generic and becomes my own method.

---

### Point 1 — Use per-STATE optimality, not the whole-network gradient  ★ the key idea

**What:** The naive energy measures "is the whole network's gradient zero?" A better idea:
measure "is the policy optimal **at each state separately**?" — using the **advantage function**
my critic already computes.

Intuition: a policy is optimal at a state when its **advantage** is zero for the actions it
takes and ≤ 0 for the actions it avoids. That's a clean math condition (a "KKT condition"),
and my critic already gives me the advantage. So the energy becomes: *"how much is this
optimality condition violated, added up over states."*

**Problem with the naive whole-network gradient:** it's too coarse. The gradient can be zero
*on average* while the policy is terrible at individual states — the good states and bad states
**cancel out**. So "gradient = 0" can lie.

**Fix:** Build the energy from **per-state advantage conditions** instead. This is:
- finer-grained (can't hide bad states by averaging),
- built from the critic I already have,
- and *visibly different* from generic Hamiltonian descent → this is my novelty.

---

### Point 2 — The critic must join the energy

**What:** The per-state advantage condition only makes sense if the critic is *accurate*. So I
add "is the critic self-consistent?" (the Bellman condition) as another piece of the energy.

**Problem:** If the critic is wrong, the advantage is wrong, so the optimality condition I'm
measuring is measuring garbage.

**Fix:** Make the total energy = team-optimality + adversary-optimality + **critic-consistency**
+ belief-accuracy. One single scalar containing everything. Minimize that.

---

### Point 3 — The noise floor (my biggest worry) has a clean fix

**What:** I worried that `‖gradient‖²` computed from *sampled* data never actually reaches zero.

**Problem (the "variance floor"):** When you square a noisy estimate, you accidentally include
its *noise*, not just its true value. Formally: `average(g²) = (true g)² + noise`. So even at the
perfect solution where the true gradient is zero, the measured energy stays stuck above zero
because of sampling noise. The certificate ("energy → 0") would be unreachable.

**Fix:** Compute the gradient on **two independent batches of data** and multiply them together
(`g₁ · g₂`) instead of squaring one. Because the two batches are independent, their noise
**cancels**, giving an honest estimate of the true squared gradient that *can* reach zero.

**Caveats to remember:**
- On small batches this can dip slightly negative (so "energy ≥ 0" only holds on average). Probably fine, worth a proof.
- The higher-order gradient terms are subtle — there's a known tool for this in multi-agent RL
  called **DiCE / Loaded DiCE** (Foerster et al.) that I'll likely need to cite/use.
- Point 1 helps here too: building the energy from *critic outputs* instead of noisy sampled
  gradients moves much of the noise into the (more controllable) critic error.

---

## 3. Extra notes / suggestions

### Note A — Entropy regularization (a needed ingredient)
My energy method technically needs a "strong concavity" condition to work. The RL-native way to
get it: **add entropy** to the objective (encourage the policy to stay a bit random). This:
- makes each per-state problem nicely behaved (strongly concave),
- keeps solutions in the safe interior (away from messy edge cases where softmax gradients die),
- can be slowly reduced ("annealed") back toward the true un-regularized solution,
- also helps avoid getting stuck at bad solutions.
**Takeaway:** plan to include entropy from the start.

### Note B — The scaling fear goes away (good news)
The original energy paper is expensive: cost grows **quadratically with the number of players**
(every player differentiates through every other). But in my setup, **the whole team shares one
policy network and there's one adversary** — so it's effectively just a **2-player game** no
matter how many agents N I have. The expensive scaling never hits me. Worth stating as a
structural *advantage* of my c-MARL setting.

### Note C — The reviewer question I MUST answer: "why not just use extragradient?"
There's a cheaper family of methods (**optimistic / extragradient**) that *also* kills limit
cycles, using only first-order gradients (no expensive Hessians).

**My honest answer for why energy is still worth it:**
- extragradient only has guarantees in "nice" (convex) settings; mine isn't nice,
- it gives **no convergence certificate** — nothing observable tells you you've arrived,
- it has **no constraint machinery**.
- My energy gives all three: a certificate (energy → 0), an ε-equilibrium bound, and proper
  handling of the action-probability (simplex) constraints.

**Critical practical advice:** run **extragradient as a baseline from day one** on the toy
game. If it quietly fixes my cycling in the real environments, I need to lead with the
"certificate + constraints" story, not "GDA cycles."

### Note D — Safety net against bad solutions
`‖gradient‖²`-type energies can be attracted to **team-pessimal** stationary points (solutions
where the team *loses* but the math is still "stationary"). Cheap insurance: descend the energy
**plus a small amount of the normal GDA update** ("consensus optimization"). Build this in from
the start.

---

## 4. Where to start (my plan)

The Fable model voted to write the precise per-state energy first (the elegant core). But I
think the smarter first move is to **validate cheaply**:

1. On the **toy matrix game**, first run:
   - plain two-timescale GDA (reproduce the cycling — confirm the disease exists),
   - **extragradient** (does the cheap method already fix it?),
   - **naive Hamiltonian descent** (does the simple energy help at all?).
2. Only if the simple versions aren't enough, build the **sophisticated per-state
   advantage-complementarity energy** (Point 1) with entropy (Note A) and the two-batch
   estimator (Point 3).

**Reason:** everything the Fable model proposed is *more complex* than my current code. If I
build the fancy version first and it fails, I won't know which piece broke. Validate the simple
things first, add sophistication only when needed.

---

## 5. One-line summary of the whole plan

> My max-min training cycles instead of converging. Turn it into minimizing a single "energy"
> that is zero only at the robust equilibrium — built from **per-state advantage conditions**
> (using my critic), made noise-proof with a **two-batch estimator**, stabilized with
> **entropy**, and benchmarked honestly against **extragradient**. The payoff: stable training
> plus a convergence *certificate* the current method can't give.
