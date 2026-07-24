import numpy as np

def sample_types_batched(cfg, B):
    """(B, N) theta. Each env independently: prob p_no_adv all-coop, else one random adversary."""
    theta = np.zeros((B, cfg.N), dtype=int)
    has_adv = np.random.rand(B) >= cfg.p_no_adv
    adv_idx = np.random.randint(cfg.N, size=B)
    theta[has_adv, adv_idx[has_adv]] = 1
    return theta
    
def random_rollout_batched(env, cfg, B):
    """B episodes, all agents act randomly. Returns (mean per-step reward per env (B,), theta (B,N))."""
    theta = sample_types_batched(cfg, B)          # (B, N)
    total = np.zeros(B, dtype=np.float32)
    for _ in range(cfg.ep_len):
        actions = np.random.randint(cfg.n_actions, size=(B, cfg.N))   # (B, N)
        _, r, done = env.step(actions)            # r: (B,)
        total += r
        if done:
            break
    return total / cfg.ep_len, theta

def sample_batched(logits_np):
    """logits_np:(B, n_actions) -> actions:(B,)"""
    m = logits_np - logits_np.max(axis=1, keepdims=True)
    probs = np.exp(m); probs /= probs.sum(axis=1, keepdims=True)
    r = np.random.rand(probs.shape[0], 1)
    return (probs.cumsum(axis=1) > r).argmax(axis=1).astype(int)

def belief_features_batched(state_oh, prev_actions, cfg):
    """(B, n_states) + (B, N) -> (B, feat_dim)."""
    B = state_oh.shape[0]
    a_oh = np.eye(cfg.n_actions, dtype=np.float32)[prev_actions].reshape(B, -1)  # (B, N*A)
    return np.concatenate([state_oh, a_oh], axis=1).astype(np.float32)