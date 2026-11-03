"""Honest robustness evaluation.

Freeze the trained team policy and belief, then train a FRESH adversary
(with its own fresh critic) against them. The team never updates, so the
reward the fresh adversary converges to measures how exploitable the team is.

Report robust_reward (mean of reward_adv over the last `window` iterations)
and worst_reward (the minimum in that window).
"""

import copy

import torch

from cmarl.agents.belief import Belief
from cmarl.agents.critic import VCritic
from cmarl.agents.policy import AdversaryPolicy, TeamPolicy
from cmarl.algorithms.base import reward_stats
from cmarl.algorithms.gda import GDAConfig
from cmarl.byzantine.wrapper import ByzantineEnv
from cmarl.losses.ppo import entropy_bonus, gae, normalize, ppo_policy_loss, value_loss
from cmarl.rollout import collect
from cmarl.utils.ego import egocentric


def _freeze(module: torch.nn.Module) -> torch.nn.Module:
    """Copy a module and make the copy untrainable. The original is left untouched."""
    frozen = copy.deepcopy(module)
    frozen.requires_grad_(False)
    return frozen


def evaluate_robustness(env: ByzantineEnv, team: TeamPolicy, belief: Belief,
                        n_iters: int = 1000, adversary_belief_dim: int = 0,
                        config: GDAConfig = GDAConfig(), window: int = 100,
                        log_every: int = 100) -> dict:
    team = _freeze(team)
    belief = _freeze(belief)

    # Fresh attacker: new adversary, new critic, new optimizers.
    adversary = AdversaryPolicy(env.spec, belief_dim=adversary_belief_dim)
    critic = VCritic(env.spec)
    opt_adv = torch.optim.Adam(adversary.parameters(), lr=config.lr_adv)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=config.lr_critic)

    def step(opt, loss):
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(opt.param_groups[0]["params"], config.max_grad_norm)
        opt.step()

    curve = []                                             # reward_adv at every iteration
    for it in range(1, n_iters + 1):
        batch = collect(env, team, adversary, belief)
        B, T, N = batch.actions.shape

        # Critic inputs (same as GDA): per-agent state, egocentric types, time.
        state = batch.state.unsqueeze(2).expand(-1, -1, N, -1)
        theta_ego = egocentric(batch.theta).unsqueeze(1).expand(-1, T, -1, -1)
        time = (torch.arange(T) / T).view(1, T, 1, 1).expand(B, T, N, 1)
        adv_mask = (batch.theta == 1).unsqueeze(1)          # (B, 1, N)

        with torch.no_grad():
            value = critic(state, theta_ego, time)
            adv, ret = gae(batch.reward, value, config.gamma, config.lam)
            adv = normalize(adv, adv_mask)                  # statistics over adversary steps only
            z, _ = belief(batch.feat)

        adv_inputs = (batch.obs, z) if adversary_belief_dim > 0 else (batch.obs,)
        for _ in range(config.ppo_epochs):
            logp, ent = adversary.evaluate(batch.actions, *adv_inputs)
            loss = (ppo_policy_loss(logp, batch.logp, adv, adv_mask, config.clip, sign=-1.0)
                    - config.entropy_coef * entropy_bonus(ent, adv_mask))
            step(opt_adv, loss)

        step(opt_critic, value_loss(critic(state, theta_ego, time), ret))

        stats = reward_stats(batch)
        curve.append(stats.get("reward_adv", float("nan")))
        if it % log_every == 0:
            print(f"attack iter {it:5d} | reward_adv {curve[-1]:.3f} | "
                  f"reward_clean {stats.get('reward_clean', float('nan')):.3f}")

    curve = torch.tensor(curve)
    last = curve[-window:]
    return {
        "curve": curve,
        "robust_reward": last.nanmean().item(),
        "worst_reward": last[~last.isnan()].min().item(),
    }