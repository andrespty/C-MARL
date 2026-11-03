"""Two-agent toy coordination game.

Two states, two actions per agent, both agents fully observe the state.

    State s0 (XNOR): reward 1 if the agents AGREE,  else 0.
    State s1 (XOR) : reward 1 if the agents DIFFER, else 0.

Transitions:
    s0: agree  -> s1,  differ -> s0
    s1: agree  -> s0,  differ -> s1

Robust value: if one agent is an adversary, it can hold the team to an
expected reward of 0.5 per step by playing uniformly at random, so 0.5 is the
best achievable reward under attack. Clean optimum is 1.0.
"""

import torch
import torch.nn.functional as F

from cmarl.environments.base import EnvSpec, StepResult, VecEnv


class ToyGame(VecEnv):

    def __init__(self, num_envs: int, episode_length: int = 100, device: str = "cpu"):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.spec = EnvSpec(
            n_agents=2,
            n_actions=2,
            obs_dim=2,        # one-hot state
            state_dim=2,      # one-hot state
            episode_length=episode_length,
        )
        self.t = 0
        self.state = torch.zeros(num_envs, dtype=torch.long, device=self.device)  # (B,) ints in {0, 1}

    def reset(self) -> StepResult:
        self.t = 0
        # Random initial state in each env.
        self.state = torch.randint(0, 2, (self.num_envs,), device=self.device)
        zeros = torch.zeros(self.num_envs, device=self.device)
        done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return StepResult(obs=self._obs(), state=self._state(), reward=zeros, done=done)

    def step(self, actions: torch.Tensor) -> StepResult:
        # actions: (B, 2) long
        agree = actions[:, 0] == actions[:, 1]       # (B,) bool
        in_s0 = self.state == 0                      # (B,) bool

        # Reward: agree in s0, differ in s1.
        reward = torch.where(in_s0, agree, ~agree).float()

        # Transition: from s0, agree -> s1; from s1, agree -> s0; differing stays put.
        # Agreeing flips the state, differing keeps it.
        self.state = torch.where(agree, 1 - self.state, self.state)

        self.t += 1
        done = torch.full((self.num_envs,), self.t >= self.spec.episode_length,
                          dtype=torch.bool, device=self.device)
        return StepResult(obs=self._obs(), state=self._state(), reward=reward, done=done)

    # ----- helpers -----

    def _state(self) -> torch.Tensor:
        """Global state as one-hot: (B, 2)."""
        return F.one_hot(self.state, num_classes=2).float()

    def _obs(self) -> torch.Tensor:
        """Each agent sees the full state: (B, N, 2)."""
        return self._state().unsqueeze(1).expand(-1, self.spec.n_agents, -1)