import numpy as np

def generate_types(cfg, B):
    """(B, N) theta. Each env independently: prob p_no_adv all-coop, else one random adversary."""
    theta = np.zeros((B, cfg.N), dtype=int)
    has_adv = np.random.rand(B) >= cfg.p_no_adv
    adv_idx = np.random.randint(cfg.N, size=B)
    theta[has_adv, adv_idx[has_adv]] = 1
    return theta

def choose_action(logits_np):
    """logits_np:(B, n_actions) -> actions:(B,)"""
    m = logits_np - logits_np.max(axis=1, keepdims=True)
    probs = np.exp(m); probs /= probs.sum(axis=1, keepdims=True)
    r = np.random.rand(probs.shape[0], 1)
    return (probs.cumsum(axis=1) > r).argmax(axis=1).astype(int)

def construct_belief_input(state_oh, prev_actions, cfg):
    """(B, n_states) + (B, N) -> (B, feat_dim)."""
    B = state_oh.shape[0]
    a_oh = np.eye(cfg.n_actions, dtype=np.float32)[prev_actions].reshape(B, -1)  # (B, N*A)
    return np.concatenate([state_oh, a_oh], axis=1).astype(np.float32)