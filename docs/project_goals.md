# Project Goals — Staged Plan

A checklist so I don't get overwhelmed. Each stage produces something concrete and de-risks
the next one.

---

## Where I am right now

**End of "build it," start of "measure it."**

**Done:**
- Full working implementation of the EIR-MAPPO mechanism from scratch (team policy, adversary,
  critic, belief network)
- Batched + optimized training loop (~0.4s/iter, down from 14s)
- Reproduced the core result: belief accuracy → 99%, clean coordination → 0.99
- Found and fixed a real bug (the **egocentric belief** — each agent needs its belief about its
  *partner*, not a shared global belief, or it can't specialize its defense)
- **Discovered the limit cycle** — reward-under-attack oscillates instead of converging.
  This is my most valuable finding so far: it's the motivation for everything downstream.

**Not done:**
- No `reward(undefended)` floor → my numbers aren't yet interpretable
- Single-seed results → the 0.995 I saw might be luck
- No honest evaluation (freeze team, train fresh adversary)

**Risk to be aware of:** I have a *working system with unvalidated numbers*. It's tempting to
build forward (energy function, k-adversaries) on results I haven't verified. Stage 0 closes
that gap. Don't skip it.

---

## Stage 0 — Lock down what I have  (1–2 days)

- [ ] Add `reward(undefended)`: train a vanilla policy (no belief, no adversarial training),
      attack it, record the number. **This is the floor — I can't claim anything without it.**
- [ ] Add stability logging: std and min of `reward(adv)` over the last 50 iterations
- [ ] Run 3–5 seeds of the current method, record mean ± std

**Deliverable:** an honest baseline table. I'll know exactly how bad the cycling is and whether
my defense actually bought anything.

---

## Stage 1 — The cheap check  (1 day)

- [ ] Implement extragradient (a lookahead step in the existing update functions)
- [ ] Run on the toy, same seeds

**Deliverable:** does the cheap fix already solve cycling? **This determines my whole pitch.**
If extragradient fixes it, the energy method's story must be "certificate + constraints," not
"we fix cycling."

---

## Stage 2 — Minimal energy method  (1 week)

- [ ] Change critic: output `Q(s,·)` (vector of size `n_actions`) instead of scalar `V`
- [ ] Implement the soft residual `r(s,a) = Q − τ·log π − V_soft` for team and adversary
- [ ] Implement the Bellman residual
- [ ] Total energy = sum of squared residuals; minimize with Adam

**Deliverable:** does energy → 0? Does reward stabilize? First real test of the method.

---

## Stage 3 — Harden it  (1 week, only if Stage 2 works)

- [ ] Add entropy annealing (τ decreasing over training)
- [ ] Add consensus term (small GDA component) to avoid bad stationary points
- [ ] Multi-seed comparison: GDA vs extragradient vs energy

**Deliverable:** the methods comparison table for a paper.

---

## Stage 4 — Proper evaluation  (3–4 days)

- [ ] Freeze the trained team, train a **fresh** adversary from scratch against it
- [ ] Measure reward under that fresh attack (the honest robustness number)

**Deliverable:** publishable evaluation protocol matching the reference papers.
Note: my current `reward(adv)` is measured against the *co-trained* adversary, which is
optimistic. This stage is what makes the number credible.

---

## Stage 5 — My actual contribution: k adversaries  (2–3 weeks)

- [ ] Extend `sample_types_batched` to k independent adversaries
- [ ] Verify the factored belief works (it already outputs N sigmoids — mostly free)
- [ ] Sweep k/N, plot robustness and belief accuracy vs. k

**Deliverable:** the tolerance-threshold result. **This is the paper's core.**

---

## If I'm overwhelmed, do only this

**Stage 0 and Stage 1.** That's 2–3 days and it tells me almost everything:
- whether my current results are even meaningful (Stage 0)
- whether I need the energy method at all (Stage 1)

Everything after depends on those answers. Don't build Stage 2+ until I know.

---

## Reality check to remember

**Stage 5 is the actual novel contribution.** Stages 1–4 are *training machinery* — valuable,
but instrumental. If time gets tight, a solid k-adversary result with vanilla two-timescale
training is a better paper than perfect optimization with no new science.

**Don't let the optimization rabbit hole eat the research.**
