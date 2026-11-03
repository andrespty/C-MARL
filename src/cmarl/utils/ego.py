"""Egocentric re-indexing: reorder per-agent vectors so slot 0 is "me"."""

import torch


def egocentric(x: torch.Tensor) -> torch.Tensor:
    """Give every agent its own rolled copy of x.

    x   : (..., N, ...)  indexed by agent along dim -1, e.g. theta (B, N)
    out : (..., N, N)    out[..., i, j] = x[..., (i + j) % N]
          row i is agent i's view: slot 0 = agent i, slot 1 = agent i+1, ...
    """
    N = x.shape[-1]
    idx = (torch.arange(N).unsqueeze(1) + torch.arange(N).unsqueeze(0)) % N   # (N, N)
    return x[..., idx.to(x.device)]