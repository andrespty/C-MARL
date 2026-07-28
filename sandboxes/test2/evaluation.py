# evaluation.py
import numpy as np
import tensorflow as tf
from dataclasses import replace

from networks.policy_adv import build_adversary, update_adversary
from networks.value_net import build_value_net, update_value_net
from rollout import run_episodes, take_gru_step
from utils import compute_advantages_batched

def evaluate_robustness(game, trained_agents, belief_step, n_iters=2000,
                        hidden=128, lr_adv=1e-3, lr_value=5e-4, log_every=100):
    """
    Honest robustness evaluation:
      - FREEZE the trained team policy + belief
      - Train a FRESH adversary (with its own fresh value net) from scratch against it
      - Return the attack curve: team reward as the new adversary gets stronger

    The final/plateau reward is the real robustness number.
    """
    # fresh attacker: new adversary net + new value net + new optimizers
    eval_agents = replace(
        trained_agents,
        adversary=build_adversary(game, hidden),
        value=build_value_net(game, hidden),
        opt_adv=tf.keras.optimizers.legacy.Adam(lr_adv),
        opt_value=tf.keras.optimizers.legacy.Adam(lr_value),
    )
    # policy, belief, belief_cell, belief_head are carried over UNCHANGED (frozen)
    # belief_step still valid: it closes over the same belief_cell/head objects

    curve = []
    for it in range(1, n_iters + 1):
        traj = run_episodes(game, eval_agents, belief_step, hidden)
        G, adv = compute_advantages_batched(traj, eval_agents, game)   # your existing fn

        update_adversary(traj, adv, eval_agents, game)
        update_value_net(traj, G, eval_agents, game)   # attacker's own critic learns
        # NOTE: update_policy and update_belief are NEVER called -> team is frozen

        if it % log_every == 0:
            ev = run_episodes(game, eval_agents, belief_step, hidden)
            ep_r    = ev['reward'].mean(axis=1)
            has_adv = ev['theta'][:, 0, :].sum(axis=1) > 0
            r_adv   = ep_r[has_adv].mean()  if has_adv.any()   else float('nan')
            r_clean = ep_r[~has_adv].mean() if (~has_adv).any() else float('nan')
            curve.append((it, r_adv, r_clean))
            print(f"attack iter {it:5d} | reward(adv) {r_adv:.3f} | reward(clean) {r_clean:.3f}")

    curve = np.array(curve)
    final_window = curve[-5:, 1]          # last 5 checkpoints
    return dict(
        curve=curve,
        robust_reward_mean=final_window.mean(),
        robust_reward_min=final_window.min(),   # honest worst case
        clean_reward=curve[-1, 2],
    )