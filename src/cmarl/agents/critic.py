"""Centralized critics (training only).

Both see the global state and the TRUE types, in egocentric order
(theta_ego[..., 0] = this agent's own type). Inputs are per agent:

    state     : (..., state_dim)    e.g. (B, T, N, state_dim), the global state repeated per agent
    theta_ego : (..., N)            e.g. (B, T, N, N), from cmarl.utils.ego.egocentric(theta)

    VCritic : -> (...,)            V(s, theta)
    QCritic : -> (..., n_actions)  Q_i(s, theta, a_i) for every action of agent i
"""

from typing import Sequence

import torch
import torch.nn as nn

from cmarl.environments.base import EnvSpec
from cmarl.nets.mlp import MLP


class VCritic(nn.Module):

    def __init__(self, spec: EnvSpec, hidden_sizes: Sequence[int] = (128, 128), activations=nn.ReLU):
        super().__init__()
        self.net = MLP(spec.state_dim + spec.n_agents + 1, 1, hidden_sizes, activations)   # +1 = time

    def forward(self, state, theta_ego, time):
        # time: (..., 1), fraction of the episode elapsed, t / T
        x = torch.cat([state, theta_ego.float(), time], dim=-1)
        return self.net(x).squeeze(-1)

    

class QCritic(nn.Module):

    def __init__(self, spec: EnvSpec, hidden_sizes: Sequence[int] = (128, 128), activations=nn.ReLU):
        super().__init__()
        self.net = MLP(spec.state_dim + spec.n_agents, spec.n_actions, hidden_sizes, activations)

    def forward(self, state: torch.Tensor, theta_ego: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, theta_ego.float()], dim=-1)
        return self.net(x)                                   # (..., n_actions)

    # ----- helpers on a computed q -----

    @staticmethod
    def taken(q: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Q of the actions actually played: q (..., A), actions (...,) -> (...,)."""
        return q.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

    @staticmethod
    def value(q: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """V = sum_a pi(a) Q(a): (..., A) -> (...,)."""
        return (probs * q).sum(-1)

    @staticmethod
    def advantage(q: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """A(a) = Q(a) - V: (..., A)."""
        return q - QCritic.value(q, probs).unsqueeze(-1)