"""Faithful EIR-MAPPO training algorithm: naive two-timescale gradient descent-
ascent (GDA) policy gradient. Adversary updates every iteration (fast
timescale); team (value/belief/policy) updates every `team_interval`
iterations (slow timescale). This is the reference this project's later
optimization variants (extragradient, energy method -- see docs/project_goals.md
Stages 1-3) get compared against, so avoid editing this file once it's
validated -- add a new algorithms/*.py instead."""
import time
from core.rollout import run_episodes
from core.utils import compute_advantages_batched, compute_gae
from networks.policy import update_policy
from networks.adversary import update_adversary
from networks.value_net import update_value_net

def train(env_cls, agents, belief_step, belief_module, B=1000, n_iters=1000,
          team_interval=1, log_every=50, hidden=128):
    hist = {"reward_adv": [], "reward_clean": [], "belief_acc": []}
    t0 = time.time()
    game = env_cls(B)
    for it in range(1, n_iters + 1):
        traj = run_episodes(game, agents, belief_step, hidden)
        G, adv = compute_gae(traj, agents, game)
        update_adversary(traj, adv, agents, game)
        if it % team_interval == 0:
            update_value_net(traj, G, agents, game)
            belief_module.update_belief(traj, agents, game)
            for _ in range(5):                          # PPO epochs
                update_policy(traj, adv, agents, game, belief_module, hidden)
        if it % log_every == 0:
            game2 = env_cls(B)
            ev = run_episodes(game2, agents, belief_step, hidden)
            ep_reward = ev['reward'].mean(axis=1)
            has_adv = ev['theta'][:, 0, :].sum(axis=1) > 0
            ra = ep_reward[has_adv].mean()  if has_adv.any()   else float('nan')
            rc = ep_reward[~has_adv].mean() if (~has_adv).any() else float('nan')
            acc = belief_module.belief_accuracy(ev, game2)

            hist["reward_adv"].append(ra); hist["reward_clean"].append(rc)
            hist["belief_acc"].append(acc)
            print(f"iter {it:5d} | reward(adv) {ra:.3f} | reward(clean) {rc:.3f} "
                  f"| belief acc {acc:.3f} | {time.time()-t0:.0f}s")
    return hist
