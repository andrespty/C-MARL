"""This project's own variant: factored (N-sigmoid, amortized-hidden-state)
belief + vanilla two-timescale GDA. Same algorithm as run_baseline.py, only
the belief axis is swapped -- this is the design meant to generalize to
k>1 adversaries (docs/context_handoff.md, Stage 5). Diff its numbers against
run_baseline.py's frozen EIR-MAPPO reproduction, not against ad-hoc runs.
"""
import os
import sys

_TEST2_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.abspath(os.path.join(_TEST2_ROOT, "..", ".."))
for _p in (_TEST2_ROOT, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import tensorflow as tf

from game_environments.ToyGame import ToyGame
from core.agents import build_agents
from core.rollout import take_belief_step
from algorithms.gda_baseline import train
from evaluation.honest_eval import evaluate_robustness
from beliefs import factored_sigmoid as belief_module


def main(seed=0, B=5000, n_iters=10000, team_interval=10, log_every=100,
         hidden=128, eval_B=128, eval_n_iters=2000):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    cfg = ToyGame(eval_B)
    agents = build_agents(cfg, belief_module, hidden=hidden)
    belief_step = take_belief_step(agents, belief_module)

    hist = train(ToyGame, agents, belief_step, belief_module,
                 B=B, n_iters=n_iters, team_interval=team_interval,
                 log_every=log_every, hidden=hidden)

    eval_hist = evaluate_robustness(cfg, agents, belief_step, n_iters=eval_n_iters,
                                     hidden=hidden, log_every=log_every)
    return agents, hist, eval_hist


if __name__ == "__main__":
    main()
