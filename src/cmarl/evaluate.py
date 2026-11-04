"""Honest robustness evaluation: freeze the team and belief, train a FRESH
adversary and critic against them with the same GDA code (freeze_team=True)."""

import copy
from dataclasses import replace

import torch

from cmarl.agents.belief import Belief
from cmarl.agents.critic import VCritic
from cmarl.agents.policy import AdversaryPolicy, TeamPolicy
from cmarl.algorithms.gda import GDA, GDAConfig
from cmarl.byzantine.wrapper import ByzantineEnv, corrupt

def _freeze(module: torch.nn.Module) -> torch.nn.Module:
    frozen = copy.deepcopy(module)
    frozen.requires_grad_(False)
    return frozen


def evaluate_robustness(env: ByzantineEnv, team: TeamPolicy, belief: Belief,
                        n_iters: int = 1000, config: GDAConfig = GDAConfig(),
                        hidden_sizes=(128, 128), window: int = 100, log_every: int = 100) -> dict:
    team, belief = _freeze(team), _freeze(belief)
    attacker = GDA(
        team=team,
        adversary=AdversaryPolicy(env.spec, hidden_sizes=hidden_sizes),
        belief=belief,
        critic=VCritic(env.spec, belief.policy_dim, hidden_sizes=hidden_sizes),
        config=replace(config, freeze_team=True, team_interval=1),
    )
    history = attacker.train(env, n_iters, log_every)

    curve = torch.tensor([h.get("reward_adv", float("nan")) for h in history])
    last = curve[-window:]
    return {
        "curve": curve,
        "robust_reward": last.nanmean().item(),
        "worst_reward": last[~last.isnan()].min().item(),
    }

@torch.no_grad()
def analytical_robustness(env: ByzantineEnv, team: TeamPolicy, belief: Belief) -> float:
    """Toy game only: reward per step against the white-box worst-case adversary
    (ToyGame.worst_case_actions), averaged over making each agent the adversary.
    This is the official code's checkpoint metric. Max 0.5 = team plays 50/50 under attack.
    """
    game = env.env                                       # bypass the prior: theta is fixed below
    B, N = game.num_envs, game.spec.n_agents
    total = 0.0
    for adv_id in range(N):
        theta = torch.zeros(B, N, dtype=torch.long, device=game.device)
        theta[:, adv_id] = 1
        r = game.reset()
        h = belief.initial_state(B, N, device=game.device)
        prev_actions, rewards = None, []
        for _ in range(game.spec.episode_length):
            z, _, h = belief.step(belief.features(r.obs, prev_actions, theta), h)
            dist = team.distribution(r.obs, z)
            adv_actions = game.worst_case_actions(dist.probs)
            executed = corrupt(dist.sample(), adv_actions, theta)
            r = game.step(executed)
            rewards.append(r.reward)
            prev_actions = executed
        total += torch.stack(rewards).mean().item()
    return total / N