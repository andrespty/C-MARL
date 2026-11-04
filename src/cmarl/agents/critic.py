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
    """V per agent from the same inputs as the team policy: [obs | belief | agent ID]."""

    def __init__(self, spec: EnvSpec, belief_dim: int,
                 hidden_sizes: Sequence[int] = (128, 128), activations=nn.ReLU):
        super().__init__()
        self.n_agents = spec.n_agents
        self.net = MLP(spec.obs_dim + belief_dim + spec.n_agents, 1, hidden_sizes, activations)

    def forward(self, obs: torch.Tensor, belief: torch.Tensor) -> torch.Tensor:
        ids = torch.eye(self.n_agents, device=obs.device).expand(*obs.shape[:-1], self.n_agents)
        return self.net(torch.cat([obs, belief, ids], dim=-1)).squeeze(-1)

    

class QCritic(nn.Module):
    """Q per agent and per action, given the TRUE types (training only).

    Input : [obs | egocentric types theta_ego (N) | agent ID (N)]
    Output: Q_i(s, theta, a_i) for every action of agent i: (..., n_actions).
    The agent axis of obs must be second to last, e.g. obs (B, T, N, obs_dim).
    """

    def __init__(self, spec: EnvSpec, hidden_sizes: Sequence[int] = (128, 128), activations=nn.ReLU):
        super().__init__()
        self.n_agents = spec.n_agents
        self.net = MLP(spec.obs_dim + 2 * spec.n_agents + 1, spec.n_actions, hidden_sizes, activations)

    def forward(self, obs: torch.Tensor, theta_ego: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # time: (..., 1), fraction of the episode elapsed. Needed because returns depend on steps left.
        ids = torch.eye(self.n_agents, device=obs.device).expand(*obs.shape[:-1], self.n_agents)
        return self.net(torch.cat([obs, theta_ego.float(), ids, time], dim=-1))
    
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