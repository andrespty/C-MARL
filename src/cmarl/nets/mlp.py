"""Generic multilayer perceptron."""

from typing import Sequence

import torch
import torch.nn as nn


class MLP(nn.Module):
    """in_dim -> [Linear -> act_1] -> ... -> [Linear -> act_L] -> Linear -> out_act -> out_dim

    Args:
        in_dim, out_dim : input and output sizes.
        hidden_sizes    : width of each hidden layer, e.g. (128, 64).
        activations     : one activation class per hidden layer, e.g. (nn.ReLU, nn.Tanh),
                          or a single class used for every hidden layer.
        out_activation  : activation after the last layer, or None for raw outputs
                          (logits, Q-values). E.g. nn.Softplus for nonnegative outputs.

    Works on any leading batch shape: (..., in_dim) -> (..., out_dim).
    """

    def __init__(self, in_dim: int, out_dim: int,
                 hidden_sizes: Sequence[int] = (128, 128),
                 activations: type[nn.Module] | Sequence[type[nn.Module]] = nn.ReLU,
                 out_activation: type[nn.Module] | None = None):
        super().__init__()

        # A single class -> repeat it for every hidden layer.
        if isinstance(activations, type):
            activations = [activations] * len(hidden_sizes)
        if len(activations) != len(hidden_sizes):
            raise ValueError(f"got {len(activations)} activations for {len(hidden_sizes)} hidden layers")

        layers = []
        prev = in_dim
        for h, act in zip(hidden_sizes, activations):
            layers += [nn.Linear(prev, h), act()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        if out_activation is not None:
            layers.append(out_activation())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)