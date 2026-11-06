"""Paper evaluation protocol (EIR-MAPPO, Sec. 4 / Fig. 6 diagonal), for one method.

For each training seed: train the method, then for EACH agent i freeze the team
and train a fresh PPO adversary that always controls agent i. Report the team's
episode reward under each attack (max 50 on Toy). 5 seeds x 2 agents = 10 numbers.

Usage:
    python experiments/paper_protocol.py energy          # all seeds, one after another
    python experiments/paper_protocol.py energy 3        # only seed 3 (run seeds in parallel)
    python experiments/paper_protocol.py energy summary  # mean ± 95% CI from saved results
Methods: energy, gda
"""

import json
import os
import sys

import torch

from cmarl.agents.belief import CategoricalBelief
from cmarl.agents.critic import QCritic, VCritic
from cmarl.agents.policy import AdversaryPolicy, TeamPolicy
from cmarl.algorithms.base import reward_stats
from cmarl.algorithms.energy import Energy, EnergyConfig
from cmarl.algorithms.gda import GDA, GDAConfig
from cmarl.byzantine.priors import FixedAdversary, SingleAdversary
from cmarl.byzantine.wrapper import ByzantineEnv
from cmarl.environments.toy import ToyGame
from cmarl.evaluate import evaluate_robustness
from cmarl.rollout import collect

T = 100                      # episode length
H = (128, 128, 128)          # hidden sizes
NUM_ENVS = 40
SEEDS = [0, 1, 2, 3, 4]
TRAIN_ITERS = 2_000
ATTACK_ITERS = 1_000         # iterations to train each fresh adversary
RESULTS_DIR = "results"

ENERGY_CONFIG = EnergyConfig(residual="soft", tau=0.2, selection_coef=0.01,
                             selection_decay_iters=1000, explore_eps=0.05)
GDA_CONFIG = GDAConfig(team_interval=10)
ATTACK_CONFIG = GDAConfig()  # the same PPO attacker for every method


def make_env(prior) -> ByzantineEnv:
    return ByzantineEnv(ToyGame(num_envs=NUM_ENVS, episode_length=T), prior)


def train(method: str, seed: int):
    torch.manual_seed(seed)
    env = make_env(SingleAdversary(n_agents=2))          # paper prior: 0.5 clean, 0.25 per agent
    belief = CategoricalBelief(env.spec, env.prior)
    team = TeamPolicy(env.spec, belief_dim=belief.policy_dim, hidden_sizes=H)
    adversary = AdversaryPolicy(env.spec, hidden_sizes=H)
    if method == "energy":
        algo = Energy(team, adversary, belief, QCritic(env.spec, hidden_sizes=H), ENERGY_CONFIG)
    elif method == "gda":
        algo = GDA(team, adversary, belief,
                   VCritic(env.spec, belief_dim=belief.policy_dim, hidden_sizes=H), GDA_CONFIG)
    else:
        raise ValueError(f"unknown method {method}")
    algo.train(env, n_iters=TRAIN_ITERS, log_every=100)
    return algo


def evaluate(algo) -> dict:
    # Clean performance: no adversary in any episode.
    clean_env = make_env(SingleAdversary(n_agents=2, p_no_adv=1.0))
    clean = reward_stats(collect(clean_env, algo.team, algo.adversary, algo.belief))["reward_clean"]

    # Non-oblivious attack on each agent separately.
    attacked = []
    for agent in range(2):
        attack_env = make_env(FixedAdversary(n_agents=2, agent=agent))
        result = evaluate_robustness(attack_env, algo.team, algo.belief, n_iters=ATTACK_ITERS,
                                     config=ATTACK_CONFIG, hidden_sizes=H)
        attacked.append(result["robust_reward"] * T)           # per step -> per episode
    return {"clean_episode_reward": clean * T, "attacked_episode_reward": attacked}


def run_seed(method: str, seed: int) -> None:
    algo = train(method, seed)
    result = {"method": method, "seed": seed, **evaluate(algo)}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"{method}_seed{seed}.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(result)


def summary(method: str) -> None:
    attacked, clean = [], []
    for seed in SEEDS:
        path = os.path.join(RESULTS_DIR, f"{method}_seed{seed}.json")
        if not os.path.exists(path):
            print(f"missing {path}")
            continue
        with open(path) as f:
            r = json.load(f)
        attacked += r["attacked_episode_reward"]
        clean.append(r["clean_episode_reward"])
        print(f"seed {seed}: clean {r['clean_episode_reward']:.2f} | attacked {r['attacked_episode_reward']}")

    def mean_ci(xs):
        x = torch.tensor(xs)
        ci = 1.96 * x.std() / len(x) ** 0.5 if len(x) > 1 else torch.tensor(float("nan"))
        return x.mean().item(), ci.item()

    m, ci = mean_ci(attacked)
    print(f"\n{method}: attacked {m:.2f} ± {ci:.2f}  over {len(attacked)} attacks  (paper EIR-MAPPO: 44.14, max 50)")
    m, ci = mean_ci(clean)
    print(f"{method}: clean    {m:.2f} ± {ci:.2f}  over {len(clean)} seeds  (max 100)")


if __name__ == "__main__":
    method = sys.argv[1]
    arg = sys.argv[2] if len(sys.argv) > 2 else None
    if arg == "summary":
        summary(method)
    elif arg is not None:
        run_seed(method, int(arg))
    else:
        for seed in SEEDS:
            run_seed(method, seed)
        summary(method)