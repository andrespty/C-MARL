"""Base class for training algorithms.

An algorithm owns the shared networks (team, adversary, belief), adds its own
critic and optimizers, and implements one method:

    update(batch) -> dict of scalar logs

Everything else (the training loop, saving, recomputing beliefs) is shared here.
"""

import copy
from abc import ABC, abstractmethod
from typing import Callable

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

    def train(self, env: ByzantineEnv, n_iters: int, log_every: int = 50,
              teacher_forcing: bool = False,
              eval_fn: Callable[["Algorithm"], float] | None = None,
              eval_every: int = 25) -> list[dict[str, float]]:
        """
        teacher_forcing : in the first half of training, with probability decaying 1 -> 0,
                          the policies and critic see the true types instead of the belief.
        eval_fn(algo)   : robustness score. Every `eval_every` iterations it is computed and
                          the best-scoring weights are kept in self.best_state (checkpoint
                          selection). Restore them with load_best().
        (Both as in the official EIR-MAPPO code.)
        """
        history = []
        self.best_score, self.best_state = -float("inf"), None
        for it in range(1, n_iters + 1):
            teacher_prob = max(0.0, 1 - (it - 1) / (n_iters / 2)) if teacher_forcing else 0.0
            batch = collect(env, self.team, self.adversary, self.belief, teacher_prob)
            logs = self.update(batch)
            logs.update(reward_stats(batch))
            if hasattr(env.prior, "record") and "reward_adv" in logs:
                env.prior.record(logs["reward_adv"])
            if eval_fn is not None and it % eval_every == 0:
                logs["eval_score"] = eval_fn(self)
                if logs["eval_score"] > self.best_score:
                    self.best_score = logs["eval_score"]
                    self.best_state = copy.deepcopy(self.state_dict())
            logs["iter"] = it
            history.append(logs)
            if it % log_every == 0:
                print(" | ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in logs.items()))
        return history

    # ----- saving -----

    def state_dict(self) -> dict:
        return {name: m.state_dict() for name, m in self.modules().items()}

    def load_state_dict(self, state: dict) -> None:
        for name, m in self.modules().items():
            m.load_state_dict(state[name])

    def load_best(self) -> None:
        """Restore the weights with the best eval_fn score seen during train()."""
        if self.best_state is not None:
            self.load_state_dict(self.best_state)

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