"""Generic GRU block with sequence and single-step modes.

Leading batch dims are arbitrary, e.g. (B, N, T, in_dim): each (b, n) is an
independent sequence. Internally they are flattened because nn.GRU only
accepts (batch, T, in_dim).

Hidden state h has shape (num_layers, *lead, hidden_dim).
"""

import torch
import torch.nn as nn


class GRU(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(in_dim, hidden_dim, num_layers=num_layers, batch_first=True)

    def initial_state(self, *lead: int, device: str | torch.device = "cpu") -> torch.Tensor:
        """Zero hidden state for a batch with leading shape `lead`, e.g. initial_state(B, N)."""
        return torch.zeros(self.num_layers, *lead, self.hidden_dim, device=device)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Full sequence.

        x : (*lead, T, in_dim)
        h : (num_layers, *lead, hidden_dim) or None (zeros)
        returns
            out : (*lead, T, hidden_dim)   hidden state at every step
            h_n : (num_layers, *lead, hidden_dim)   last hidden state
        """
        *lead, T, in_dim = x.shape
        x_flat = x.reshape(-1, T, in_dim)                                   # (batch, T, in_dim)
        h_flat = None if h is None else h.reshape(self.num_layers, -1, self.hidden_dim)

        out, h_n = self.gru(x_flat, h_flat)

        out = out.reshape(*lead, T, self.hidden_dim)
        h_n = h_n.reshape(self.num_layers, *lead, self.hidden_dim)
        return out, h_n

    def step(self, x: torch.Tensor, h: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """One timestep.

        x : (*lead, in_dim)
        returns out (*lead, hidden_dim), h_next (num_layers, *lead, hidden_dim)
        """
        out, h_next = self.forward(x.unsqueeze(-2), h)    # add a T=1 axis
        return out.squeeze(-2), h_next