"""Byzantine wrapper: adds hidden types and action corruption to any VecEnv.

    reset()                    -> samples theta ~ prior (fixed for the episode)
    step(team_actions, adv_actions)
        executed[b, i] = adv_actions[b, i]   if theta[b, i] == 1
                         team_actions[b, i]  otherwise

This is the Byzantine perturbation P^alpha: compromised agents' actions are
replaced by the adversary's; cooperative agents' actions pass through.

The underlying env never sees theta. It only receives executed actions.
"""

from dataclasses import dataclass

import torch

from cmarl.byzantine.priors import TypePrior
from cmarl.environments.base import StepResult, VecEnv


@dataclass
class ByzantineStep(StepResult):
    """StepResult plus the information the Byzantine layer adds."""
    theta: torch.Tensor = None          # (B, N) long, true types (constant within an episode)
    executed: torch.Tensor = None       # (B, N) long, actions actually executed this step (None at reset)


def corrupt(team_actions: torch.Tensor, adv_actions: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Apply the Byzantine perturbation. All tensors (B, N)."""
    return torch.where(theta == 1, adv_actions, team_actions)


class ByzantineEnv:

    def __init__(self, env: VecEnv, prior: TypePrior):
        assert prior.n_agents == env.n_agents, "prior and env disagree on number of agents"
        self.env = env
        self.prior = prior
        self.theta = None

    # Expose the inner env's attributes so callers don't need env.env.spec.
    @property
    def spec(self):
        return self.env.spec

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    @property
    def device(self) -> torch.device:
        return self.env.device

    def reset(self) -> ByzantineStep:
        self.theta = self.prior.sample(self.num_envs, self.device)   # (B, N)
        r = self.env.reset()
        return ByzantineStep(obs=r.obs, state=r.state, reward=r.reward, done=r.done,
                             theta=self.theta, executed=None)

    def step(self, team_actions: torch.Tensor, adv_actions: torch.Tensor) -> ByzantineStep:
        executed = corrupt(team_actions, adv_actions, self.theta)
        r = self.env.step(executed)
        return ByzantineStep(obs=r.obs, state=r.state, reward=r.reward, done=r.done,
                             theta=self.theta, executed=executed)