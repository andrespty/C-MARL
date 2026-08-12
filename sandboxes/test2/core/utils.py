import numpy as np

def generate_types(game):
    """(B, N) theta. Each env independently: prob p_no_adv all-coop, else one random adversary."""
    theta = np.zeros((game.B, game.N), dtype=int)
    has_adv = np.random.rand(game.B) >= game.p_no_adv
    adv_idx = np.random.randint(game.N, size=game.B)
    theta[has_adv, adv_idx[has_adv]] = 1
    return theta

def choose_action(logits_np):
    """logits_np:(B, n_actions) -> actions:(B,)"""
    m = logits_np - logits_np.max(axis=1, keepdims=True)
    probs = np.exp(m); probs /= probs.sum(axis=1, keepdims=True)
    r = np.random.rand(probs.shape[0], 1)
    return (probs.cumsum(axis=1) > r).argmax(axis=1).astype(int)

def construct_belief_input(state_oh, prev_actions, game):
    """(B, n_states) + (B, N) -> (B, feat_dim)."""
    B = state_oh.shape[0]
    a_oh = np.eye(game.n_actions, dtype=np.float32)[prev_actions].reshape(B, -1)  # (B, N*A)
    return np.concatenate([state_oh, a_oh], axis=1).astype(np.float32)

def compute_returns_batched(rewards, gamma):
    B, T = rewards.shape
    G = np.zeros_like(rewards, dtype=np.float32)
    running = np.zeros(B, dtype=np.float32)
    for t in reversed(range(T)):
        running = rewards[:, t] + gamma * running
        G[:, t] = running
    return G

def compute_advantages_batched(traj, agents, game):
    B, T = traj['reward'].shape
    G = compute_returns_batched(traj['reward'], game.gamma)
    state_flat = traj['state'].reshape(B * T, game.n_states)
    theta_flat = traj['theta'].reshape(B * T, game.N).astype(np.float32)   # true types — centralized baseline
    V = agents.value([state_flat, theta_flat]).numpy().reshape(B, T)
    adv = G - V
    return G, adv


def compute_gae(traj, agents, game, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation. Returns G (returns) and adv, each (B, T)."""
    B, T = traj['reward'].shape
    rewards = traj['reward']                                    # (B, T)

    # value estimate at each timestep from the critic
    state_flat  = traj['state'].reshape(B*T, game.n_states)
    theta_flat  = traj['theta'].reshape(B*T, game.N).astype(np.float32)  # critic input
    V = agents.value([state_flat, theta_flat]).numpy().reshape(B, T)     # (B, T)

    # bootstrap value for the step after the last (0 if episode ended)
    adv = np.zeros((B, T), dtype=np.float32)
    last_gae = np.zeros(B, dtype=np.float32)
    for t in reversed(range(T)):
        V_next = V[:, t+1] if t+1 < T else np.zeros(B, dtype=np.float32)
        delta = rewards[:, t] + gamma * V_next - V[:, t]        # TD error
        last_gae = delta + gamma * lam * last_gae               # GAE recursion
        adv[:, t] = last_gae
    G = adv + V                                                 # returns = adv + baseline

    # normalize advantages
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return G.astype(np.float32), adv.astype(np.float32)