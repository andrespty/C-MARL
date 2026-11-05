"""Generic multilayer perceptron (MAPPO / official EIR-MAPPO style by default)."""

from typing import Sequence

import torch
import torch.nn as nn

_GAIN_NAMES = {nn.ReLU: "relu", nn.Tanh: "tanh", nn.LeakyReLU: "leaky_relu", nn.Sigmoid: "sigmoid"}


class MLP(nn.Module):
    """[LayerNorm] -> [Linear -> act -> LayerNorm] x L -> Linear -> [out_act]

    Args:
        in_dim, out_dim : input and output sizes.
        hidden_sizes    : width of each hidden layer, e.g. (128, 128, 128).
        activations     : one activation class per hidden layer, or a single class for all.
        out_activation  : activation after the last layer, or None for raw outputs.
        layer_norm      : LayerNorm on the input and after every hidden layer.
        orthogonal_init : orthogonal weights and zero biases. Hidden layers use the
                          activation's recommended gain (sqrt(2) for ReLU).
        out_gain        : gain of the last layer. 0.01 for policies -> near-uniform initial actions.

    Works on any leading batch shape: (..., in_dim) -> (..., out_dim).
    """

    def __init__(self, in_dim: int, out_dim: int,
                 hidden_sizes: Sequence[int] = (128, 128),
                 activations: type[nn.Module] | Sequence[type[nn.Module]] = nn.ReLU,
                 out_activation: type[nn.Module] | None = None,
                 norm: type[nn.Module] | None = nn.LayerNorm,   # nn.LayerNorm, FlatBatchNorm, or None
                 orthogonal_init: bool = True,
                 out_gain: float = 1.0):
        super().__init__()

        if isinstance(activations, type):
            activations = [activations] * len(hidden_sizes)
        if len(activations) != len(hidden_sizes):
            raise ValueError(f"got {len(activations)} activations for {len(hidden_sizes)} hidden layers")

        layers = [norm(in_dim)] if norm is not None else []
        prev = in_dim
        for h, act in zip(hidden_sizes, activations):
            gain = nn.init.calculate_gain(_GAIN_NAMES[act]) if act in _GAIN_NAMES else 1.0
            layers += [self._linear(prev, h, gain, orthogonal_init), act()]
            if norm is not None:
                layers.append(norm(h))
            prev = h
        layers.append(self._linear(prev, out_dim, out_gain, orthogonal_init))
        if out_activation is not None:
            layers.append(out_activation())

        self.net = nn.Sequential(*layers)

    @staticmethod
    def _linear(in_dim: int, out_dim: int, gain: float, orthogonal: bool) -> nn.Linear:
        layer = nn.Linear(in_dim, out_dim)
        if orthogonal:
            nn.init.orthogonal_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)