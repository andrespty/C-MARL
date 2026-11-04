"""Two-timescale GDA with PPO, following the official EIR-MAPPO implementation.

Per update(batch):
    1. beliefs z (no grad) -> critic V(obs, z, id) -> GAE advantages,
       normalized over cooperative steps only
    2. each PPO epoch: team step (team iterations only), then adversary step
    3. critic: clipped Huber loss on normalized returns (team iterations only)
    4. belief: cross-entropy (every iteration)

Team iterations: iteration % team_interval == 0.
freeze_team=True trains only adversary + critic (used by evaluate.py).
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from cmarl.agents.belief import Belief
from cmarl.agents.critic import VCritic
from cmarl.agents.policy import AdversaryPolicy, TeamPolicy
from cmarl.algorithms.base import Algorithm
from cmarl.batch import Batch
from cmarl.losses.ppo import clipped_value_loss, entropy_bonus, gae, normalize, ppo_policy_loss
from cmarl.utils.value_norm import ValueNorm


@dataclass
class GDAConfig:
    lr_team: float = 5e-4
    lr_adv: float = 5e-4
    lr_critic: float = 5e-4
    lr_belief: float = 5e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.05
    ppo_epochs: int = 5
    critic_epochs: int = 5
    belief_epochs: int = 5
    entropy_coef: float = 0.01
    adv_entropy_coef: float = 0.01
    team_interval: int = 1
    max_grad_norm: float = 10.0
    huber_delta: float = 10.0
    use_value_norm: bool = True
    freeze_team: bool = False


class GDA(Algorithm):

    def __init__(self, team: TeamPolicy, adversary: AdversaryPolicy, belief: Belief,
                 critic: VCritic, config: GDAConfig = GDAConfig()):
        super().__init__(team, adversary, belief)
        self.critic = critic
        self.cfg = config
        self.iteration = 0
        self.value_norm = ValueNorm() if config.use_value_norm else None
        self.opt_team = self.make_optimizer(team, config.lr_team)
        self.opt_adv = self.make_optimizer(adversary, config.lr_adv)
        self.opt_critic = self.make_optimizer(critic, config.lr_critic)
        self.opt_belief = self.make_optimizer(belief, config.lr_belief)

    def modules(self) -> dict[str, nn.Module]:
        return {**super().modules(), "critic": self.critic}

    def _step(self, opt: torch.optim.Optimizer, loss: torch.Tensor) -> float:
        opt.zero_grad()
        loss.backward()
        params = [p for group in opt.param_groups for p in group["params"]]
        nn.utils.clip_grad_norm_(params, self.cfg.max_grad_norm)
        opt.step()
        return loss.item()

    def update(self, batch: Batch) -> dict[str, float]:
        self.iteration += 1
        cfg = self.cfg
        logs = {}

        team_iteration = self.iteration % cfg.team_interval == 0
        train_team = team_iteration and not cfg.freeze_team
        train_critic = team_iteration or cfg.freeze_team
        team_mask = (batch.theta == 0).unsqueeze(1)             # (B, 1, N)
        adv_mask = ~team_mask

        # 1. advantages
        with torch.no_grad():
            # z, _ = self.beliefs(batch)                          # (B, T, N, Z)
            z = batch.z                                         # (B, T, N, Z) exactly what the policies saw
            v_old = self.critic(batch.obs, z)                   # (B, T, N), normalized scale
            v_real = self.value_norm.denormalize(v_old) if self.value_norm else v_old
            adv, ret = gae(batch.reward, v_real, cfg.gamma, cfg.lam)
            adv = normalize(adv, adv_mask if cfg.freeze_team else team_mask)

        # 2. actors, interleaved per epoch
        for _ in range(cfg.ppo_epochs):
            if train_team:
                logp, ent = self.team.evaluate(batch.actions, batch.obs, z)
                loss = (ppo_policy_loss(logp, batch.logp, adv, team_mask, cfg.clip, sign=+1.0)
                        - cfg.entropy_coef * entropy_bonus(ent, team_mask))
                logs["loss_team"] = self._step(self.opt_team, loss)

            logp, ent = self.adversary.evaluate(batch.actions, *self.adversary_inputs(batch.obs, z))
            loss = (ppo_policy_loss(logp, batch.logp, adv, adv_mask, cfg.clip, sign=-1.0)
                    - cfg.adv_entropy_coef * entropy_bonus(ent, adv_mask))
            logs["loss_adv"] = self._step(self.opt_adv, loss)

        # 3. critic
        if train_critic:
            if self.value_norm:
                self.value_norm.update(ret)
                target = self.value_norm.normalize(ret)
            else:
                target = ret
            for _ in range(cfg.critic_epochs):
                loss = clipped_value_loss(self.critic(batch.obs, z), v_old, target,
                                          cfg.clip, cfg.huber_delta)
                logs["loss_critic"] = self._step(self.opt_critic, loss)

        # 4. belief
        if train_team:
            for _ in range(cfg.belief_epochs):
                _, out = self.beliefs(batch)
                logs["loss_belief"] = self._step(self.opt_belief, self.belief.loss(out, batch.theta))
            logs["belief_acc"] = self.belief.accuracy(out.detach(), batch.theta).item()

        return logs