"""Base class for training algorithms.

An algorithm owns the shared networks (team, adversary, belief), adds its own
critic and optimizers, and implements one method:

    update(batch) -> dict of scalar logs

Everything else (the training loop, saving, recomputing beliefs) is shared here.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from cmarl.agents.belief import Belief
from cmarl.agents.policy import AdversaryPolicy, TeamPolicy
from cmarl.batch import Batch
from cmarl.byzantine.wrapper import ByzantineEnv
from cmarl.rollout import collect


class Algorithm(ABC):

    def __init__(self, team: TeamPolicy, adversary: AdversaryPolicy, belief: Belief):
        self.team = team
        self.adversary = adversary
        self.belief = belief

    # ----- what subclasses implement -----

    @abstractmethod
    def update(self, batch: Batch) -> dict[str, float]:
        """One training step on a batch. Returns logs, e.g. {"loss_team": 0.1}."""

    def modules(self) -> dict[str, nn.Module]:
        """All networks of this algorithm. Subclasses add their critic(s)."""
        return {"team": self.team, "adversary": self.adversary, "belief": self.belief}

    # ----- shared helpers -----

    def beliefs(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Recompute beliefs over the whole batch WITH gradients.

        Returns z (B, T, N, Z) for the policies and out (B, T, N, O) for the belief loss.
        """
        return self.belief(batch.feat)

    def adversary_inputs(self, obs: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Arguments for the adversary: (obs,) or (obs, belief) depending on how it was built."""
        return (obs, z) if self.adversary.belief_dim > 0 else (obs,)

    @staticmethod
    def make_optimizer(module: nn.Module, lr: float) -> torch.optim.Optimizer:
        return torch.optim.Adam(module.parameters(), lr=lr)

    # ----- training loop -----

    def train(self, env: ByzantineEnv, n_iters: int, log_every: int = 50) -> list[dict[str, float]]:
        history = []
        for it in range(1, n_iters + 1):
            batch = collect(env, self.team, self.adversary, self.belief)
            logs = self.update(batch)
            logs.update(reward_stats(batch))
            logs["iter"] = it
            history.append(logs)
            if it % log_every == 0:
                print(" | ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}"
                                 for k, v in logs.items()))
        return history

    # ----- saving -----

    def state_dict(self) -> dict:
        return {name: m.state_dict() for name, m in self.modules().items()}

    def load_state_dict(self, state: dict) -> None:
        for name, m in self.modules().items():
            m.load_state_dict(state[name])

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path))


def reward_stats(batch: Batch) -> dict[str, float]:
    """Mean per-step reward on clean episodes and on attacked episodes."""
    ep_reward = batch.reward.mean(dim=1)                     # (B,)
    attacked = batch.has_adversary
    stats = {}
    if (~attacked).any():
        stats["reward_clean"] = ep_reward[~attacked].mean().item()
    if attacked.any():
        stats["reward_adv"] = ep_reward[attacked].mean().item()
    return stats