import numpy as np
import tensorflow as tf

def compute_returns_batched(rewards, gamma):
    B, T = rewards.shape
    G = np.zeros_like(rewards, dtype=np.float32)
    running = np.zeros(B, dtype=np.float32)
    for t in reversed(range(T)):
        running = rewards[:, t] + gamma * running
        G[:, t] = running
    return G

def compute_advantages_batched(traj, agents, cfg):
    B, T = traj['reward'].shape
    G = compute_returns_batched(traj['reward'], cfg.gamma)
    state_flat = traj['state'].reshape(B*T, cfg.n_states)
    theta_flat = traj['theta'].reshape(B*T, cfg.N).astype(np.float32)   # true types — centralized baseline
    V = agents.value([state_flat, theta_flat]).numpy().reshape(B, T)
    adv = G - V
    return G, adv
