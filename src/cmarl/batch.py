"""Container for one batch of rollouts. Shape convention: (B, T, N, ...)."""

from dataclasses import dataclass, fields

import torch


@dataclass
class Batch:
    obs: torch.Tensor          # (B, T, N, obs_dim)  each agent's observation
    state: torch.Tensor        # (B, T, state_dim)   global state (critics)
    feat: torch.Tensor         # (B, T, N, F)        belief features (to recompute beliefs with gradients)
    actions: torch.Tensor      # (B, T, N) long      executed actions
    team_actions: torch.Tensor # (B, T, N) long      actions proposed by the team policy
    adv_actions: torch.Tensor  # (B, T, N) long      actions proposed by the adversary
    logp: torch.Tensor         # (B, T, N)           log-prob of the executed action under the policy that chose it
    reward: torch.Tensor       # (B, T)              shared reward
    done: torch.Tensor         # (B, T) bool
    theta: torch.Tensor        # (B, N) long         true types (constant within an episode)

    @property
    def num_envs(self) -> int:
        return self.reward.shape[0]

    @property
    def horizon(self) -> int:
        return self.reward.shape[1]

    @property
    def has_adversary(self) -> torch.Tensor:
        """(B,) bool: episodes with at least one compromised agent."""
        return self.theta.sum(-1) > 0

    def to(self, device) -> "Batch":
        return Batch(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})