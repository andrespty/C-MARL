import numpy as np
import tensorflow as tf
from Cooperative_MARL.sandboxes.test1.misc import sample_types_batched, belief_features_batched, sample_batched

def make_belief_step(agents):
    """Build the compiled single-step GRU fn bound to these agents' weights."""
    @tf.function(reduce_retracing=True)
    def belief_step(feat_t, h_prev):
        out, new_states = agents.belief_cell(feat_t, states=[h_prev])
        return agents.belief_head(out), new_states[0]
    return belief_step

def run_episodes_batched(env_cls, cfg, B, agents, belief_step, hidden=128):
    """Run B episodes in parallel. Returns dict of (B, T, ...) arrays."""
    env = env_cls(cfg, B)
    theta = sample_types_batched(cfg, B)
    prev_actions = np.zeros((B, cfg.N), dtype=int)
    h = tf.zeros((B, hidden))
    S_state, S_belief, S_actions, S_reward, S_feat = [], [], [], [], []

    for t in range(cfg.ep_len):
        obs_all = env._obs()
        state_oh = obs_all[:, 0, :]

        # Compute belief based on observations
        f = belief_features_batched(state_oh, prev_actions, cfg)
        b_tf, h = belief_step(tf.constant(f), h)
        b_t = b_tf.numpy()

        # --- cooperative actions: one batched policy call over all agents ---
        b_ego_all  = np.stack([np.roll(b_t, -i, axis=1) for i in range(cfg.N)], axis=1)  # (B,N,N)
        state_rep  = np.repeat(state_oh, cfg.N, axis=0)          # (B*N, n_states)
        b_ego_flat = b_ego_all.reshape(B*cfg.N, cfg.N)           # (B*N, N)
        c_logits   = agents.policy([state_rep, b_ego_flat]).numpy()  # (B*N, A)
        a_coop     = sample_batched(c_logits).reshape(B, cfg.N)      # (B, N)

        # --- adversary actions: one batched call, one batched sample ---
        a_logits = agents.adversary(obs_all.reshape(B*cfg.N, cfg.n_states)).numpy()  # (B*N, A)
        a_adv    = sample_batched(a_logits).reshape(B, cfg.N)                        # (B, N)

        # --- apply corruption ---
        actions = np.where(theta == 1, a_adv, a_coop)

        _, r, done = env.step(actions)
        S_state.append(state_oh); S_belief.append(b_t)
        S_actions.append(actions.copy()); S_reward.append(r); S_feat.append(f)
        prev_actions = actions
        if done: break

    stack = lambda L: np.stack(L, axis=1)
    return dict(
        state=stack(S_state), belief=stack(S_belief),
        actions=stack(S_actions), reward=stack(S_reward), feat=stack(S_feat),
        theta=np.repeat(theta[:, None, :], len(S_reward), axis=1),
    )