"""PPO building blocks. Pure functions: tensors in, tensors out.

Shapes: B = envs, T = time, N = agents.
    reward      : (B, T)
    value       : (B, T) or (B, T, N)     critic output per step (per agent if egocentric)
    logp_*      : (B, T, N)
    adv, mask   : broadcastable to (B, T, N)

Episodes have a fixed length and end at T, so the value after the last step is 0.
"""

import torch


def masked_mean(x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Mean of x over entries where mask == 1 (all entries if mask is None)."""
    if mask is None:
        return x.mean()
    mask = mask.float().expand_as(x)
    return (x * mask).sum() / mask.sum().clamp(min=1.0)


def normalize(x: torch.Tensor, mask: torch.Tensor | None = None, eps: float = 1e-8) -> torch.Tensor:
    """Zero mean, unit std (statistics over masked entries)."""
    mean = masked_mean(x, mask)
    std = masked_mean((x - mean) ** 2, mask).sqrt()
    return (x - mean) / (std + eps)


def _match(reward: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Reshape reward (B, T) so it broadcasts against `like` (B, T, ...)."""
    return reward.reshape(*reward.shape, *[1] * (like.dim() - 2))


def discounted_returns(reward: torch.Tensor, gamma: float) -> torch.Tensor:
    """G_t = r_t + gamma * G_{t+1}, with G_T = 0.  (B, T) -> (B, T)."""
    G = torch.zeros_like(reward)
    running = torch.zeros_like(reward[:, 0])
    for t in reversed(range(reward.shape[1])):
        running = reward[:, t] + gamma * running
        G[:, t] = running
    return G


def gae(reward: torch.Tensor, value: torch.Tensor, gamma: float, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation.

        delta_t = r_t + gamma * V_{t+1} - V_t          (V_T = 0)
        A_t     = delta_t + gamma * lam * A_{t+1}

    value : (B, T) or (B, T, N).  Returns (advantages, returns), both shaped like value.
    returns = A + V is the critic's regression target.
    """
    value = value.detach()
    r = _match(reward, value)
    next_value = torch.cat([value[:, 1:], torch.zeros_like(value[:, :1])], dim=1)   # V_{t+1}, 0 at the end
    delta = r + gamma * next_value - value

    adv = torch.zeros_like(value)
    running = torch.zeros_like(value[:, 0])
    for t in reversed(range(value.shape[1])):
        running = delta[:, t] + gamma * lam * running
        adv[:, t] = running
    return adv, adv + value


def ppo_policy_loss(logp_new: torch.Tensor, logp_old: torch.Tensor, adv: torch.Tensor,
                    mask: torch.Tensor, clip: float = 0.2, sign: float = 1.0) -> torch.Tensor:
    """Clipped PPO surrogate, returned as a loss to MINIMIZE.

    sign = +1 : maximize the team's advantage (team policy)
    sign = -1 : minimize it (adversary)
    mask      : 1 on the steps this policy is responsible for
                (theta == 0 for the team, theta == 1 for the adversary)
    """
    a = sign * adv.detach()
    ratio = torch.exp(logp_new - logp_old.detach())
    surrogate = torch.min(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a)
    return -masked_mean(surrogate, mask)


def value_loss(value: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Mean squared error to the (detached) target."""
    return masked_mean((value - target.detach()) ** 2, mask)


def entropy_bonus(entropy: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Average entropy. Subtract coef * this from the loss to encourage exploration."""
    return masked_mean(entropy, mask)