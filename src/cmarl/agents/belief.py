"""Belief networks: infer the other agents' types from the history.

Every agent runs the same network on its own egocentric features:
    [own obs | everyone's previous executed actions, ego order, one-hot | own type]
A GRU accumulates the history; a head turns the GRU state into a belief.

Shapes (convention: time before agents, like the rest of the library):
    step     : feat (B, N, F)    -> z (B, N, Z), out (B, N, O), h_next
    forward  : feat (B, T, N, F) -> z (B, T, N, Z), out (B, T, N, O)
    theta    : (B, N) true types, constant within an episode

    z   = what the policy conditions on (GRU hidden state or belief probabilities)
    out = raw head output (logits), used by the losses

Variants:
    CategoricalBelief : softmax over all type profiles in the prior's support (exact, K classes)
    FactoredBelief    : one sigmoid per agent (approximate, N outputs)

Both assume a permutation-symmetric prior (true for SingleAdversary and IndependentK),
so the same support can be used in every agent's egocentric frame.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from cmarl.byzantine.priors import TypePrior
from cmarl.environments.base import EnvSpec
from cmarl.nets.gru import GRU
from cmarl.utils.ego import egocentric


class Belief(nn.Module):

    def __init__(self, spec: EnvSpec, prior: TypePrior, hidden_dim: int = 128,
                 policy_input: str = "hidden"):
        super().__init__()
        assert policy_input in ("hidden", "probs")
        self.spec = spec
        self.hidden_dim = hidden_dim
        self.policy_input = policy_input
        self.feat_dim = spec.obs_dim + spec.n_agents * spec.n_actions + 1
        self.gru = GRU(self.feat_dim, hidden_dim)
        # Buffer = a tensor that belongs to the module (moves with .to(device), saved in
        # state_dict) but is NOT a trainable parameter.
        self.register_buffer("support", prior.support.clone())          # (K, N) long
        self.head: nn.Module = None                                      # set by subclass

    # ----- sizes -----

    @property
    def out_dim(self) -> int:
        raise NotImplementedError

    @property
    def policy_dim(self) -> int:
        """Size of z, i.e. the belief_dim to give the policies."""
        return self.hidden_dim if self.policy_input == "hidden" else self.out_dim

    # ----- inputs -----

    def features(self, obs: torch.Tensor, prev_actions: torch.Tensor | None,
                 own_type: torch.Tensor) -> torch.Tensor:
        """Build one step of features.

        obs          : (B, N, obs_dim)
        prev_actions : (B, N) long executed actions of the previous step, or None at t = 0
        own_type     : (B, N) long, theta
        returns      : (B, N, feat_dim)
        """
        B, N = obs.shape[:2]
        A = self.spec.n_actions
        if prev_actions is None:
            acts = torch.zeros(B, N, N * A, device=obs.device)
        else:
            acts = F.one_hot(egocentric(prev_actions), A).float()        # (B, N, N, A)
            acts = acts.flatten(start_dim=-2)                             # (B, N, N*A)
        return torch.cat([obs, acts, own_type.float().unsqueeze(-1)], dim=-1)

    def initial_state(self, B: int, N: int, device=None) -> torch.Tensor:
        return self.gru.initial_state(B, N, device=device)

    # ----- running the network -----

    def step(self, feat: torch.Tensor, h: torch.Tensor):
        """One timestep (rollout). Returns (z, out, h_next)."""
        hid, h_next = self.gru.step(feat, h)              # (B, N, H)
        out = self.head(hid)
        return self._policy_features(hid, out), out, h_next

    def forward(self, feat_seq: torch.Tensor):
        """Whole episode (training). feat_seq (B, T, N, F). Returns (z, out)."""
        x = feat_seq.transpose(1, 2)                      # (B, N, T, F): GRU wants time second to last
        hid, _ = self.gru(x)                              # (B, N, T, H)
        hid = hid.transpose(1, 2)                         # back to (B, T, N, H)
        out = self.head(hid)
        return self._policy_features(hid, out), out

    def _policy_features(self, hid: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        return hid if self.policy_input == "hidden" else self.probs(out)

    # ----- implemented by subclasses -----

    def probs(self, out: torch.Tensor) -> torch.Tensor:
        """Head output -> belief probabilities."""
        raise NotImplementedError

    def loss(self, out: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Supervised loss against the true types. out (B, T, N, O), theta (B, N)."""
        raise NotImplementedError

    def profile_probs(self, out: torch.Tensor, own_type: torch.Tensor) -> torch.Tensor:
        """Belief over type profiles: (..., K), row k = P(my egocentric theta = support[k]).

        own_type must broadcast to out.shape[:-1], e.g. (B, 1, N) for out (B, T, N, O).
        Profiles inconsistent with the agent's own (known) type get probability 0.
        """
        raise NotImplementedError

    def accuracy(self, out: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # ----- shared helper -----

    def _own_type_mask(self, own_type: torch.Tensor) -> torch.Tensor:
        """True where support[k, 0] != own type: (..., K)."""
        return self.support[:, 0] != own_type.unsqueeze(-1)


class CategoricalBelief(Belief):
    """Softmax over the K type profiles of the prior. Default policy input: the probabilities."""

    def __init__(self, spec: EnvSpec, prior: TypePrior, hidden_dim: int = 128,
                 policy_input: str = "probs"):
        super().__init__(spec, prior, hidden_dim, policy_input)
        self.head = nn.Linear(hidden_dim, self.support.shape[0])

    @property
    def out_dim(self) -> int:
        return self.support.shape[0]

    def probs(self, out):
        return out.softmax(-1)

    def labels(self, theta: torch.Tensor) -> torch.Tensor:
        """Index of each agent's egocentric theta in the support: (B, N) long."""
        ego = egocentric(theta)                                       # (B, N, N)
        match = (ego.unsqueeze(-2) == self.support).all(-1)           # (B, N, K)
        return match.float().argmax(-1)

    def loss(self, out, theta):
        labels = self.labels(theta).unsqueeze(1).expand(out.shape[:-1])   # (B, T, N)
        return F.cross_entropy(out.reshape(-1, self.out_dim), labels.reshape(-1))

    def profile_probs(self, out, own_type):
        return out.masked_fill(self._own_type_mask(own_type), float("-inf")).softmax(-1)

    def accuracy(self, out, theta):
        labels = self.labels(theta).unsqueeze(1)                      # (B, 1, N)
        return (out.argmax(-1) == labels).float().mean()


class FactoredBelief(Belief):
    """One sigmoid per egocentric slot: P(agent in slot j is compromised).
    Slot 0 (self) is known, so it is excluded from loss and accuracy.
    Default policy input: the GRU hidden state.
    """

    def __init__(self, spec: EnvSpec, prior: TypePrior, hidden_dim: int = 128,
                 policy_input: str = "hidden"):
        super().__init__(spec, prior, hidden_dim, policy_input)
        self.head = nn.Linear(hidden_dim, spec.n_agents)

    @property
    def out_dim(self) -> int:
        return self.spec.n_agents

    def probs(self, out):
        return out.sigmoid()

    def _targets(self, theta, out):
        return egocentric(theta).float().unsqueeze(1).expand_as(out)   # (B, T, N, N)

    def loss(self, out, theta):
        target = self._targets(theta, out)
        return F.binary_cross_entropy_with_logits(out[..., 1:], target[..., 1:])

    def profile_probs(self, out, own_type):
        # Product of independent Bernoullis over slots 1..N-1, restricted to the support.
        s = self.support.float()                                      # (K, N)
        log_p = F.logsigmoid(out).unsqueeze(-2)                       # (..., 1, N)  log P(compromised)
        log_q = F.logsigmoid(-out).unsqueeze(-2)                      # (..., 1, N)  log P(cooperative)
        log_b = (s * log_p + (1 - s) * log_q)[..., 1:].sum(-1)       # (..., K)
        return log_b.masked_fill(self._own_type_mask(own_type), float("-inf")).softmax(-1)

    def accuracy(self, out, theta):
        target = self._targets(theta, out)
        return ((out[..., 1:] > 0) == target[..., 1:].bool()).float().mean()