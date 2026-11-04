"""Running mean/std normalizer for value targets.
Same math as the official code's `PopArt` class (common/popart.py)."""

import torch


class ValueNorm:

    def __init__(self, beta: float = 0.99999, epsilon: float = 1e-5):
        self.beta, self.epsilon = beta, epsilon
        self.mean_acc, self.sq_acc, self.debias = 0.0, 0.0, 0.0

    def _stats(self) -> tuple[float, float]:
        d = max(self.debias, self.epsilon)
        mean = self.mean_acc / d
        var = max(self.sq_acc / d - mean ** 2, 1e-2)
        return mean, var ** 0.5

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        b = self.beta
        self.mean_acc = b * self.mean_acc + (1 - b) * x.mean().item()
        self.sq_acc = b * self.sq_acc + (1 - b) * (x ** 2).mean().item()
        self.debias = b * self.debias + (1 - b)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean, std = self._stats()
        return (x - mean) / std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        mean, std = self._stats()
        return x * std + mean