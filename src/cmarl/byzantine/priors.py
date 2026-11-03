"""Priors over type profiles theta in {0,1}^N  (1 = compromised agent).

Every prior is defined by its support (all possible theta) and their
probabilities. Sampling is then just "pick a row of the support".

    support : (K, N) long   every theta with nonzero probability
    probs   : (K,)   float  prior probability of each row
    sample  : (B, N) long   one theta per parallel env

Enumerating the support is exact and simple. It costs sum_{m<=k} C(N, m) rows,
which is fine for the small N we use.
"""

from abc import ABC, abstractmethod
from itertools import combinations

import torch


class TypePrior(ABC):

    def __init__(self, n_agents: int):
        self.n_agents = n_agents
        self.support, self.probs = self._build()   # (K, N), (K,)

    @abstractmethod
    def _build(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (support, probs). probs must sum to 1."""

    def sample(self, num_envs: int, device: str | torch.device = "cpu") -> torch.Tensor:
        """Sample one theta per env: (B, N) long."""
        idx = torch.multinomial(self.probs, num_envs, replacement=True)  # (B,) row indices
        return self.support[idx].to(device)

    @property
    def num_profiles(self) -> int:
        return self.support.shape[0]


class SingleAdversary(TypePrior):
    """At most one adversary (the setting of EIR-MAPPO).

    With probability p_no_adv nobody is compromised; otherwise exactly one
    agent, chosen uniformly, is compromised.

    Support rows: [0,...,0], then e_1, ..., e_N  (K = N + 1).
    """

    def __init__(self, n_agents: int, p_no_adv: float = 0.5):
        self.p_no_adv = p_no_adv
        super().__init__(n_agents)

    def _build(self):
        N = self.n_agents
        support = torch.cat([
            torch.zeros(1, N, dtype=torch.long),   # nobody compromised
            torch.eye(N, dtype=torch.long),        # agent i compromised
        ])
        probs = torch.cat([
            torch.tensor([self.p_no_adv]),
            torch.full((N,), (1 - self.p_no_adv) / N),
        ])
        return support, probs


class IndependentK(TypePrior):
    """Each agent compromised independently with probability q,
    conditioned on at most k compromised agents:

        p(theta) ∝ q^|theta| (1-q)^(N-|theta|),   |theta| <= k
    """

    def __init__(self, n_agents: int, q: float, k: int):
        assert 0 <= k <= n_agents
        self.q = q
        self.k = k
        super().__init__(n_agents)

    def _build(self):
        N = self.n_agents
        rows, weights = [], []
        for m in range(self.k + 1):                       # number of adversaries
            for idx in combinations(range(N), m):         # which agents
                theta = torch.zeros(N, dtype=torch.long)
                theta[list(idx)] = 1
                rows.append(theta)
                weights.append(self.q ** m * (1 - self.q) ** (N - m))
        support = torch.stack(rows)                        # (K, N)
        probs = torch.tensor(weights)
        return support, probs / probs.sum()                # normalize (the conditioning on |theta| <= k)