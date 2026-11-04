"""Discrete-action policies.

    CategoricalPolicy : shared machinery (logits -> distribution, sample, log-probs, entropy)
    TeamPolicy        : input = [own observation | belief features]
    AdversaryPolicy   : input = own observation

All methods work on any leading batch shape, e.g. (B, N, obs_dim).
Subclasses only implement forward(*inputs) -> logits.
"""

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from cmarl.environments.base import EnvSpec
from cmarl.nets.mlp import MLP


class CategoricalPolicy(nn.Module):

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """Return action logits (..., n_actions)."""
        raise NotImplementedError

    def distribution(self, *inputs: torch.Tensor) -> Categorical:
        return Categorical(logits=self(*inputs))

    def log_probs_all(self, *inputs: torch.Tensor) -> torch.Tensor:
        """log pi(a | x) for every action: (..., n_actions). Used by the energy method."""
        return F.log_softmax(self(*inputs), dim=-1)

    def act(self, *inputs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample actions. Returns (actions (...,) long, log_probs (...,))."""
        dist = self.distribution(*inputs)
        actions = dist.probs.argmax(-1) if deterministic else dist.sample()
        return actions, dist.log_prob(actions)

    def evaluate(self, actions: torch.Tensor, *inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Log-prob of given actions and entropy, with gradients. Used by policy-gradient losses."""
        dist = self.distribution(*inputs)
        return dist.log_prob(actions), dist.entropy()


class TeamPolicy(CategoricalPolicy):
    """Cooperative policy, shared by all agents. Conditions on the belief and,
    by default, on a one-hot agent ID (needed to break symmetry between agents)."""

    def __init__(self, spec: EnvSpec, belief_dim: int,
                 hidden_sizes: Sequence[int] = (128, 128),
                 activations=nn.ReLU, use_agent_id: bool = True):
        super().__init__()
        self.n_agents = spec.n_agents
        self.use_agent_id = use_agent_id
        in_dim = spec.obs_dim + belief_dim + (spec.n_agents if use_agent_id else 0)
        self.net = MLP(in_dim, spec.n_actions, hidden_sizes, activations, out_gain=0.01)

    def forward(self, obs: torch.Tensor, belief: torch.Tensor) -> torch.Tensor:
        # obs (..., N, obs_dim): the agent axis must be second to last.
        parts = [obs, belief]
        if self.use_agent_id:
            ids = torch.eye(self.n_agents, device=obs.device)             # (N, N)
            parts.append(ids.expand(*obs.shape[:-1], self.n_agents))     # (..., N, N)
        return self.net(torch.cat(parts, dim=-1))


class AdversaryPolicy(CategoricalPolicy):
    """Adversarial policy, shared by all compromised agents.
    Input: [obs | belief (if belief_dim > 0) | own agent ID]."""

    def __init__(self, spec: EnvSpec, belief_dim: int = 0,
                 hidden_sizes: Sequence[int] = (128, 128),
                 activations=nn.ReLU, use_agent_id: bool = True):
        super().__init__()
        self.n_agents = spec.n_agents
        self.belief_dim = belief_dim
        self.use_agent_id = use_agent_id
        in_dim = spec.obs_dim + belief_dim + (spec.n_agents if use_agent_id else 0)
        self.net = MLP(in_dim, spec.n_actions, hidden_sizes, activations, out_gain=0.01)

    def forward(self, obs: torch.Tensor, belief: torch.Tensor | None = None) -> torch.Tensor:
        parts = [obs]
        if self.belief_dim > 0:
            parts.append(belief)
        if self.use_agent_id:
            ids = torch.eye(self.n_agents, device=obs.device)
            parts.append(ids.expand(*obs.shape[:-1], self.n_agents))
        return self.net(torch.cat(parts, dim=-1))