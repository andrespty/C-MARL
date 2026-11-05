"""Energy-method building blocks. Pure functions: tensors in, tensors out.

Per-state optimality of a player choosing probabilities p over actions, given
advantages A (KKT conditions on the simplex):
    p(a) >= 0,   -A(a) >= 0,   p(a) * A(a) = 0
i.e. played actions have A = 0, unplayed actions have A <= 0.

sign = +1 for a maximizer (team), -1 for a minimizer (adversary): the adversary's
advantage is -A.

Shapes: probs, q, adv : (..., n_actions).  Energies : (...,).
"""

import torch


def fischer_burmeister(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """phi(a, b) = a + b - sqrt(a^2 + b^2).  phi = 0  <=>  a >= 0, b >= 0, a * b = 0."""
    return a + b - torch.sqrt(a ** 2 + b ** 2 + eps)


def advantage(q: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """A(a) = Q(a) - sum_a' p(a') Q(a')."""
    return q - (probs * q).sum(-1, keepdim=True)


def kkt_energy(probs: torch.Tensor, q: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
    """Psi = sum_a phi(p(a), -sign * A(a))^2.   Zero iff p is a best response to q.

    (The multiplier mu = -sign * A is substituted in, so no mu network is needed.)
    sign: broadcastable to (...,), +1 team / -1 adversary.
    """
    adv = advantage(q, probs)
    phi = fischer_burmeister(probs, -sign.unsqueeze(-1) * adv)
    return (phi ** 2).sum(-1)


def soft_energy(log_probs: torch.Tensor, q: torch.Tensor, sign: torch.Tensor, tau: float) -> torch.Tensor:
    """Entropy-regularized version.  Optimal p = softmax(sign * q / tau), i.e.
        tau * log p(a) = sign * q(a) - V_tau,   V_tau = tau * logsumexp(sign * q / tau).
    Psi = sum_a (tau * log p(a) - (sign * q(a) - V_tau))^2.  Zero iff p is that softmax
    (a quantal response). Tends to a best response as tau -> 0.
    log_probs: log p from log_softmax of the logits (not log of probs: that underflows).
    """
    sq = sign.unsqueeze(-1) * q
    v_tau = tau * torch.logsumexp(sq / tau, dim=-1, keepdim=True)
    residual = tau * log_probs - (sq - v_tau)
    return (residual ** 2).sum(-1)


def belief_average(q_profiles: torch.Tensor, profile_probs: torch.Tensor) -> torch.Tensor:
    """Q averaged over type profiles with the agent's belief: what the agent can actually know.

    q_profiles    : (..., K, n_actions)  Q for each possible type profile
    profile_probs : (..., K)             belief over profiles
    returns       : (..., n_actions)
    """
    return (profile_probs.unsqueeze(-1) * q_profiles).sum(-2)


def kl_energy(log_probs: torch.Tensor, q: torch.Tensor, sign: torch.Tensor, tau: float) -> torch.Tensor:
    """Psi = tau * KL(p || softmax(sign * q / tau)): the soft exploitability gap, in reward units.
    Zero iff p is the soft best response. Errors weighted by p, so unplayed actions barely count.
    log_probs: log p from log_softmax of the logits.
    """
    log_p_star = torch.log_softmax(sign.unsqueeze(-1) * q / tau, dim=-1)
    return tau * (log_probs.exp() * (log_probs - log_p_star)).sum(-1)