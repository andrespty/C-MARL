"""Base interface for vectorized cooperative multi-agent environments.

An environment here is a plain cooperative game: it knows nothing about
types, beliefs or adversaries. Byzantine behaviour is added on top by
cmarl.byzantine.wrapper.

Shape conventions (B = number of parallel envs, N = number of agents):
    obs     : (B, N, obs_dim)   float   each agent's local observation
    state   : (B, state_dim)    float   global state (for centralized critics)
    actions : (B, N)            long    one discrete action per agent
    reward  : (B,)              float   shared team reward
    done    : (B,)              bool    episode finished
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EnvSpec:
    """Static sizes of an environment. Networks are built from this."""
    n_agents: int
    n_actions: int
    obs_dim: int
    state_dim: int
    episode_length: int


@dataclass
class StepResult:
    """What step() returns. A dataclass instead of a tuple so fields have names."""
    obs: torch.Tensor      # (B, N, obs_dim)
    state: torch.Tensor    # (B, state_dim)
    reward: torch.Tensor   # (B,)
    done: torch.Tensor     # (B,) bool


class VecEnv(ABC):
    """B parallel copies of a cooperative multi-agent game.

    Subclasses must set `self.spec`, `self.num_envs`, `self.device`
    and implement reset() and step().
    """

    spec: EnvSpec
    num_envs: int
    device: torch.device

    @abstractmethod
    def reset(self) -> StepResult:
        """Start new episodes in all B envs.

        Returns a StepResult with zero reward and done = False.
        """

    @abstractmethod
    def step(self, actions: torch.Tensor) -> StepResult:
        """Advance all B envs by one step.

        Args:
            actions: (B, N) long tensor, the actions actually executed.
        """

    # Convenience shortcuts so callers can write env.n_agents instead of env.spec.n_agents.
    @property
    def n_agents(self) -> int:
        return self.spec.n_agents

    @property
    def n_actions(self) -> int:
        return self.spec.n_actions