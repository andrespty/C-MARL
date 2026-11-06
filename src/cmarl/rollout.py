"""Collect B episodes in parallel with the current networks.

At every step t:
    1. belief features from (obs_t, executed actions_{t-1}, own type)
    2. belief GRU step -> z_t (policy input)
    3. team AND adversary propose an action for every agent
    4. the Byzantine wrapper executes the right one per agent
No gradients here: training recomputes everything it needs from the Batch.
"""

import torch
from torch.distributions import Categorical

from cmarl.agents.belief import Belief
from cmarl.agents.policy import AdversaryPolicy, TeamPolicy
from cmarl.batch import Batch
from cmarl.byzantine.wrapper import ByzantineEnv


def adversary_act(adversary: AdversaryPolicy, obs: torch.Tensor, z: torch.Tensor):
    """The adversary uses the belief only if it was built with belief_dim > 0."""
    return adversary.act(obs, z) if adversary.belief_dim > 0 else adversary.act(obs)

def sample_explore(dist: Categorical, epsilon: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample from the mixture (1 - epsilon) * pi + epsilon * uniform, so every action keeps
    being tried. Returns (actions, log-prob of those actions under the mixture)."""
    probs = dist.probs
    if epsilon > 0:
        probs = (1 - epsilon) * probs + epsilon / probs.shape[-1]
    mixed = Categorical(probs=probs)
    actions = mixed.sample()
    return actions, mixed.log_prob(actions)

@torch.no_grad()
def collect(env: ByzantineEnv, team: TeamPolicy, adversary: AdversaryPolicy, belief: Belief,
            teacher_prob: float = 0.0, epsilon: float = 0.0) -> Batch:
    """teacher_prob: probability, per step, of replacing the belief by the true types
    (teacher forcing, as in the official EIR-MAPPO code). 0 = off."""
    B, N = env.num_envs, env.spec.n_agents
    r = env.reset()
    theta = r.theta                                              # (B, N)
    h = belief.initial_state(B, N, device=env.device)
    prev_actions = None

    keys = ["obs", "state", "feat", "actions", "team_actions", "adv_actions", "logp", "reward", "done", "z"]
    traj = {k: [] for k in keys}

    for t in range(env.spec.episode_length):
        feat = belief.features(r.obs, prev_actions, theta)        # (B, N, F)
        z, _, h = belief.step(feat, h)                           # z (B, N, Z)
        if teacher_prob > 0 and torch.rand(()).item() < teacher_prob:
            z = belief.true_input(theta)                         # teacher forcing: the truth, not the belief

        adv_inputs = (r.obs, z) if adversary.belief_dim > 0 else (r.obs,)
        team_a, team_lp = sample_explore(team.distribution(r.obs, z), epsilon)        # (B, N), (B, N)
        adv_a, adv_lp = sample_explore(adversary.distribution(*adv_inputs), epsilon)

        nxt = env.step(team_a, adv_a)
        logp = torch.where(theta == 1, adv_lp, team_lp)          # log-prob of what was executed

        traj["obs"].append(r.obs)
        traj["state"].append(r.state)
        traj["feat"].append(feat)
        traj["actions"].append(nxt.executed)
        traj["team_actions"].append(team_a)
        traj["adv_actions"].append(adv_a)
        traj["logp"].append(logp)
        traj["reward"].append(nxt.reward)
        traj["done"].append(nxt.done)
        traj["z"].append(z)

        prev_actions = nxt.executed
        r = nxt
        if nxt.done.all():
            break

    stacked = {k: torch.stack(v, dim=1) for k, v in traj.items()}   # list of (B, ...) -> (B, T, ...)
    return Batch(**stacked, theta=theta)