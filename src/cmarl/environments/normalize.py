"""Observation normalization with running statistics, like Stable-Baselines' VecNormalize.
Wrap the plain game: ByzantineEnv(NormalizeObs(ToyGame(...)), prior)."""

import torch

from cmarl.environments.base import StepResult, VecEnv


class RunningMeanStd:
    """Running mean and variance per feature (parallel Welford update)."""

    def __init__(self, dim: int, device="cpu"):
        self.mean = torch.zeros(dim, device=device)
        self.var = torch.ones(dim, device=device)
        self.count = 1e-4

    def update(self, x: torch.Tensor) -> None:
        """x: (M, dim) batch of samples."""
        b_mean, b_var, b_count = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = b_mean - self.mean
        total = self.count + b_count
        self.mean = self.mean + delta * b_count / total
        m2 = self.var * self.count + b_var * b_count + delta ** 2 * self.count * b_count / total
        self.var = m2 / total
        self.count = total


class NormalizeObs(VecEnv):

    def __init__(self, env: VecEnv, clip: float = 10.0, eps: float = 1e-8):
        self.env, self.clip, self.eps = env, clip, eps
        self.spec, self.num_envs, self.device = env.spec, env.num_envs, env.device
        self.training = True                       # False: use the stats, don't update them
        self.obs_rms = RunningMeanStd(env.spec.obs_dim, env.device)
        self.state_rms = RunningMeanStd(env.spec.state_dim, env.device)

    def reset(self) -> StepResult:
        return self._normalize(self.env.reset())

    def step(self, actions: torch.Tensor) -> StepResult:
        return self._normalize(self.env.step(actions))

    def _normalize(self, r: StepResult) -> StepResult:
        if self.training:
            self.obs_rms.update(r.obs.reshape(-1, r.obs.shape[-1]))
            self.state_rms.update(r.state)
        obs = ((r.obs - self.obs_rms.mean) / torch.sqrt(self.obs_rms.var + self.eps)).clamp(-self.clip, self.clip)
        state = ((r.state - self.state_rms.mean) / torch.sqrt(self.state_rms.var + self.eps)).clamp(-self.clip, self.clip)
        return StepResult(obs=obs, state=state, reward=r.reward, done=r.done)

    def __getattr__(self, name):
        """Forward anything else (e.g. ToyGame.worst_case_actions) to the wrapped game."""
        return getattr(self.env, name)