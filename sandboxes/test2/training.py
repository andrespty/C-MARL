import numpy as np
import time
import tensorflow as tf

from utils import *
from rollout import *
from networks.policy_coop import update_policy
from networks.policy_adv import update_adversary
from networks.belief_net import update_belief
from networks.value_net import update_value_net


def train(env_cls, agents, belief_step, B=1000, n_iters=1000, team_interval=1, log_every=50):
    hist = {"reward_adv": [], "reward_clean": [], "belief_acc": []}
    t0 = time.time()
    game = env_cls(B)
    for it in range(1, n_iters + 1):
        traj = run_episodes(game, agents, belief_step)
        G, adv = compute_advantages_batched(traj, agents, game)
        update_adversary(traj, adv, agents, game)
        if it % team_interval == 0:
            update_value_net(traj, G, agents, game)
            update_belief(traj, agents, game)
            update_policy(traj, adv, agents, game)
        if it % log_every == 0:
            game2 = env_cls(B)
            ev = run_episodes(game2, agents, belief_step)
            ep_reward = ev['reward'].mean(axis=1)
            has_adv = ev['theta'][:, 0, :].sum(axis=1) > 0
            ra = ep_reward[has_adv].mean()  if has_adv.any()   else float('nan')
            rc = ep_reward[~has_adv].mean() if (~has_adv).any() else float('nan')

            # belief is (B,T,N,N) egocentric: agent i predicts theta rolled by -i
            th     = ev['theta']                                              # (B,T,N)
            th_ego = np.stack([np.roll(th, -i, axis=2) for i in range(game.N)], axis=2)  # (B,T,N,N)
            pred   = (ev['belief'] > 0.5).astype(int)
            acc    = (pred[..., 1:] == th_ego[..., 1:]).mean()               # drop self-slot 0

            hist["reward_adv"].append(ra); hist["reward_clean"].append(rc)
            hist["belief_acc"].append(acc)
            print(f"iter {it:5d} | reward(adv) {ra:.3f} | reward(clean) {rc:.3f} "
                  f"| belief acc {acc:.3f} | {time.time()-t0:.0f}s")
    return hist