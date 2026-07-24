import numpy as np
import time
import tensorflow as tf

from updates import *
from rollout import *


def train(env_cls, cfg, agents, belief_step, B=1000, n_iters=1000, team_interval=1, log_every=50):
    hist = {"reward_adv": [], "reward_clean": [], "belief_acc": []}
    t0 = time.time()
    for it in range(1, n_iters + 1):
        traj = run_episodes(env_cls, cfg, B, agents, belief_step)
        G, adv = compute_advantages_batched(traj, agents, cfg)
        update_adversary(traj, adv, agents, cfg)
        if it % team_interval == 0:
            update_critic(traj, G, agents, cfg)
            update_belief(traj, agents, cfg)
            update_policy(traj, adv, agents, cfg)
        if it % log_every == 0:
            ev = run_episodes(env_cls, cfg, B, agents, belief_step)
            ep_reward = ev['reward'].mean(axis=1)
            has_adv = ev['theta'][:, 0, :].sum(axis=1) > 0
            ra = ep_reward[has_adv].mean()  if has_adv.any()   else float('nan')
            rc = ep_reward[~has_adv].mean() if (~has_adv).any() else float('nan')

            # belief is (B,T,N,N) egocentric: agent i predicts theta rolled by -i
            th     = ev['theta']                                              # (B,T,N)
            th_ego = np.stack([np.roll(th, -i, axis=2) for i in range(cfg.N)], axis=2)  # (B,T,N,N)
            pred   = (ev['belief'] > 0.5).astype(int)
            acc    = (pred[..., 1:] == th_ego[..., 1:]).mean()               # drop self-slot 0

            hist["reward_adv"].append(ra); hist["reward_clean"].append(rc)
            hist["belief_acc"].append(acc)
            print(f"iter {it:5d} | reward(adv) {ra:.3f} | reward(clean) {rc:.3f} "
                  f"| belief acc {acc:.3f} | {time.time()-t0:.0f}s")
    return hist