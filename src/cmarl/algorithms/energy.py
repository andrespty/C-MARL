"""Energy method: find the robust equilibrium by MINIMIZING one energy that is
zero exactly at equilibrium, instead of running gradient descent-ascent.

Per update(batch):
    1. critic  : regress Q_i(s, theta, a_i) on a TD (or Monte Carlo) target
    2. energy  : E = mean over steps and agents of Psi(p_i, Q_i), with
                   p_i  = the acting policy's probabilities (team or adversary)
                   Q_i  = belief-averaged critic, DETACHED (the critic cannot cheat)
                   sign = +1 team, -1 adversary
                 Team and adversary both DESCEND E (one minimization, no ascent).
                 + selection_coef * team policy-gradient term, to prefer
                   high-reward equilibria (E is also 0 at bad ones, e.g. uniform play).
    3. belief  : cross-entropy (every iteration)
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from cmarl.agents.belief import Belief
from cmarl.agents.critic import QCritic
from cmarl.agents.policy import AdversaryPolicy, TeamPolicy
from cmarl.algorithms.base import Algorithm
from cmarl.batch import Batch
from cmarl.losses.energy import advantage, belief_average, kkt_energy, soft_energy, kl_energy
from cmarl.utils.ego import egocentric
from cmarl.losses.ppo import discounted_returns, masked_mean, normalize


@dataclass
class EnergyConfig:
    lr_policy: float = 5e-4           # team and adversary
    lr_critic: float = 5e-4
    lr_belief: float = 5e-4
    gamma: float = 0.99
    critic_epochs: int = 5
    energy_epochs: int = 5
    belief_epochs: int = 5
    residual: str = "fb"              # "fb" (KKT + Fischer-Burmeister) or "soft" (entropy-regularized)
    tau: float = 0.1                  # temperature, soft residual only
    critic_target: str = "td"         # "td" (expected SARSA) or "mc" (discounted return)
    belief_averaged: bool = True      # ex interim: average Q over the belief, not the true types
    selection_coef: float = 0.01      # team policy-gradient term (equilibrium selection)
    selection_decay_iters: int = 0    # linearly decay selection_coef to 0 over this many iterations (0 = no decay)
    max_grad_norm: float = 10.0


class Energy(Algorithm):

    def __init__(self, team: TeamPolicy, adversary: AdversaryPolicy, belief: Belief,
                 critic: QCritic, config: EnergyConfig = EnergyConfig()):
        super().__init__(team, adversary, belief)
        self.critic = critic
        self.cfg = config
        self.iteration = 0
        self.opt_policy = torch.optim.Adam(list(team.parameters()) + list(adversary.parameters()), lr=config.lr_policy)
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

    # ----- helpers -----

    def _acting_probs(self, batch: Batch) -> torch.Tensor:
        """Probabilities of the policy each agent actually uses: (B, T, N, A). With gradients."""
        p_team = self.team.distribution(batch.obs, batch.z).probs
        p_adv = self.adversary.distribution(*self.adversary_inputs(batch.obs, batch.z)).probs
        return torch.where((batch.theta == 1)[:, None, :, None], p_adv, p_team)

    def _acting_log_probs(self, batch: Batch) -> torch.Tensor:
        """log p of the policy each agent actually uses, from the logits (log_softmax):
        exact even for tiny probabilities, and the gradient never vanishes. (B, T, N, A)."""
        lp_team = self.team.log_probs_all(batch.obs, batch.z)
        lp_adv = self.adversary.log_probs_all(*self.adversary_inputs(batch.obs, batch.z))
        return torch.where((batch.theta == 1)[:, None, :, None], lp_adv, lp_team)

    @staticmethod
    def _time(batch: Batch) -> torch.Tensor:
        """Fraction of the episode elapsed, per step and agent: (B, T, N, 1)."""
        B, T, N = batch.actions.shape
        t = torch.arange(T, device=batch.obs.device) / T
        return t.view(1, T, 1, 1).expand(B, T, N, 1)

    def _q_true(self, batch: Batch) -> torch.Tensor:
        """Q with the true types: (B, T, N, A)."""
        T = batch.horizon
        theta_ego = egocentric(batch.theta).unsqueeze(1).expand(-1, T, -1, -1)
        return self.critic(batch.obs, theta_ego, self._time(batch))

    def _q_belief(self, batch: Batch) -> torch.Tensor:
        """Q averaged over the agent's belief about the types: (B, T, N, A)."""
        support = self.belief.support                                   # (K, N) egocentric profiles
        K = support.shape[0]
        obs = batch.obs.unsqueeze(0).expand(K, *batch.obs.shape)        # (K, B, T, N, O)
        theta = support.view(K, 1, 1, 1, -1).expand(*obs.shape[:-1], -1)
        time = self._time(batch).unsqueeze(0).expand(K, -1, -1, -1, -1)
        q_profiles = self.critic(obs, theta, time).permute(1, 2, 3, 0, 4)   # (B, T, N, K, A)
        _, out = self.beliefs(batch)
        b = self.belief.profile_probs(out, batch.theta.unsqueeze(1))   # (B, T, N, K)
        return belief_average(q_profiles, b)

    # ----- the update -----

    def update(self, batch: Batch) -> dict[str, float]:
        cfg = self.cfg
        logs = {}
        self.iteration += 1
        team_mask = (batch.theta == 0).unsqueeze(1)                     # (B, 1, N)
        sign = torch.where(batch.theta == 1, -1.0, 1.0).unsqueeze(1)    # (B, 1, N)

        # 1. critic
        with torch.no_grad():
            if cfg.critic_target == "mc":
                target = discounted_returns(batch.reward, cfg.gamma)
                target = target.unsqueeze(-1).expand_as(batch.actions).clone()
            else:
                p_next = self._acting_probs(batch)[:, 1:]               # π^i(·|o_{t+1})  (B, T-1, N, A)
                v_next = (p_next * self._q_true(batch)[:, 1:]).sum(-1)  # Σ_a' π Q        (B, T-1, N)
                target = batch.reward.unsqueeze(-1).expand_as(batch.actions).clone()   # r_t for each agent  (B, T, N)
                target[:, :-1] += cfg.gamma * v_next                     # last step: terminal
        for _ in range(cfg.critic_epochs):
            q_taken = self._q_true(batch).gather(-1, batch.actions.unsqueeze(-1)).squeeze(-1)
            logs["loss_critic"] = self._step(self.opt_critic, ((q_taken - target) ** 2).mean())

        # 2. energy (critic detached)
        with torch.no_grad():
            coef = cfg.selection_coef
            if cfg.selection_decay_iters > 0:
                coef *= max(0.0, 1 - self.iteration / cfg.selection_decay_iters)
            q = self._q_belief(batch) if cfg.belief_averaged else self._q_true(batch)
        for _ in range(cfg.energy_epochs):
            log_probs = self._acting_log_probs(batch)                   # (B, T, N, A), with gradients
            probs = log_probs.exp()
            
            if cfg.residual == "soft":
                psi = soft_energy(log_probs, q, sign, cfg.tau)           # (B, T, N)
            elif cfg.residual == "kl":
                psi = kl_energy(log_probs, q, sign, cfg.tau)
            else:
                psi = kkt_energy(probs, q, sign)
            energy = psi.mean()

            # equilibrium selection: push the team toward higher-value equilibria
            logp = log_probs.gather(-1, batch.actions.unsqueeze(-1)).squeeze(-1)
            a_taken = advantage(q, probs.detach()).gather(-1, batch.actions.unsqueeze(-1)).squeeze(-1)
            a_taken = normalize(a_taken, team_mask)                     # unit scale, like PPO
            selection = -masked_mean(logp * a_taken, team_mask)

            self._step(self.opt_policy, energy + coef * selection)
            logs["energy"] = energy.item()
            logs["energy_team"] = masked_mean(psi.detach(), team_mask).item()
            logs["energy_adv"] = masked_mean(psi.detach(), ~team_mask).item()

        logs["selection_coef"] = coef
        # 3. belief
        for _ in range(cfg.belief_epochs):
            _, out = self.beliefs(batch)
            logs["loss_belief"] = self._step(self.opt_belief, self.belief.loss(out, batch.theta))
        logs["belief_acc"] = self.belief.accuracy(out.detach(), batch.theta).item()
        return logs