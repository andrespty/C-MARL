"""Two-timescale gradient descent-ascent with PPO (EIR-MAPPO baseline).

Per update(batch):
    1. critic values  -> GAE advantages (per agent, egocentric critic)
    2. adversary      : PPO, sign -1, on compromised agents' steps   (every iteration)
    3. every `team_interval` iterations:
         critic : regress V onto GAE returns
         belief : cross-entropy against the true types
         team   : PPO, sign +1, on cooperative agents' steps

By default the belief is trained ONLY by cross-entropy: the policies see it as a fixed
input (detached). Set policy_grad_into_belief=True to also backprop the team's PPO loss into it.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from cmarl.agents.belief import Belief
from cmarl.agents.critic import VCritic
from cmarl.agents.policy import AdversaryPolicy, TeamPolicy
from cmarl.algorithms.base import Algorithm
from cmarl.batch import Batch
from cmarl.losses.ppo import entropy_bonus, gae, normalize, ppo_policy_loss, value_loss
from cmarl.utils.ego import egocentric


@dataclass
class GDAConfig:
    lr_team: float = 5e-4
    lr_adv: float = 1e-3              # adversary faster (two timescales)
    lr_critic: float = 5e-4
    lr_belief: float = 5e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    ppo_epochs: int = 5
    entropy_coef: float = 0.01
    team_interval: int = 1            # team updates every k iterations, adversary every iteration
    max_grad_norm: float = 0.5
    policy_grad_into_belief: bool = False


class GDA(Algorithm):

    def __init__(self, team: TeamPolicy, adversary: AdversaryPolicy, belief: Belief,
                 critic: VCritic, config: GDAConfig = GDAConfig()):
        super().__init__(team, adversary, belief)
        self.critic = critic
        self.cfg = config
        self.iteration = 0

        team_params = list(team.parameters())
        if config.policy_grad_into_belief:
            team_params += list(belief.parameters())
        self.opt_team = torch.optim.Adam(team_params, lr=config.lr_team)
        self.opt_adv = self.make_optimizer(adversary, config.lr_adv)
        self.opt_critic = self.make_optimizer(critic, config.lr_critic)
        self.opt_belief = self.make_optimizer(belief, config.lr_belief)

    def modules(self) -> dict[str, nn.Module]:
        return {**super().modules(), "critic": self.critic}

    # ----- one optimizer step -----

    def _step(self, opt: torch.optim.Optimizer, loss: torch.Tensor) -> float:
        opt.zero_grad()                                    # clear old gradients
        loss.backward()                                    # compute new ones
        params = [p for group in opt.param_groups for p in group["params"]]
        nn.utils.clip_grad_norm_(params, self.cfg.max_grad_norm)
        opt.step()                                         # apply them
        return loss.item()

    # ----- inputs shared by several losses -----

    def _critic_inputs(self, batch: Batch) -> tuple[torch.Tensor, ...]:
        """Per agent: state (B,T,N,S), egocentric types (B,T,N,N), time t/T (B,T,N,1)."""
        B, T, N = batch.actions.shape
        state = batch.state.unsqueeze(2).expand(-1, -1, N, -1)
        theta_ego = egocentric(batch.theta).unsqueeze(1).expand(-1, T, -1, -1)
        time = (torch.arange(T, device=state.device) / T).view(1, T, 1, 1).expand(B, T, N, 1)
        return state, theta_ego, time

    # ----- the update -----

    def update(self, batch: Batch) -> dict[str, float]:
        self.iteration += 1
        cfg = self.cfg
        logs = {}

        critic_in = self._critic_inputs(batch)              # CHANGED: (state, theta_ego, time)
        team_mask = (batch.theta == 0).unsqueeze(1)
        adv_mask = ~team_mask

        with torch.no_grad():
            value = self.critic(*critic_in)                 # CHANGED
            adv, ret = gae(batch.reward, value, cfg.gamma, cfg.lam)
            adv = normalize(adv)
            z_fixed, _ = self.beliefs(batch)

        # --- adversary: every iteration (fast timescale) ---
        for _ in range(cfg.ppo_epochs):
            logp, ent = self.adversary.evaluate(batch.actions, *self.adversary_inputs(batch.obs, z_fixed))
            loss = (ppo_policy_loss(logp, batch.logp, adv, adv_mask, cfg.clip, sign=-1.0)
                    - cfg.entropy_coef * entropy_bonus(ent, adv_mask))
            logs["loss_adv"] = self._step(self.opt_adv, loss)

        # --- team side: every team_interval iterations (slow timescale) ---
        if self.iteration % cfg.team_interval == 0:
            logs["loss_critic"] = self._step(
                self.opt_critic, value_loss(self.critic(*critic_in), ret))   

            _, out = self.beliefs(batch)
            logs["loss_belief"] = self._step(self.opt_belief, self.belief.loss(out, batch.theta))
            logs["belief_acc"] = self.belief.accuracy(out.detach(), batch.theta).item()

            for _ in range(cfg.ppo_epochs):
                z = self.beliefs(batch)[0] if cfg.policy_grad_into_belief else z_fixed
                logp, ent = self.team.evaluate(batch.actions, batch.obs, z)
                loss = (ppo_policy_loss(logp, batch.logp, adv, team_mask, cfg.clip, sign=+1.0)
                        - cfg.entropy_coef * entropy_bonus(ent, team_mask))
                logs["loss_team"] = self._step(self.opt_team, loss)

        return logs