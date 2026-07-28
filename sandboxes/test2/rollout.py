import numpy as np
import tensorflow as tf
from data_generation import generate_types, construct_belief_input, choose_action

def take_gru_step(agents):
    """Build the compiled single-step GRU fn bound to these agents' weights."""
    @tf.function(reduce_retracing=True)
    def belief_step(feat_t, h_prev):
        out, new_states = agents.belief_cell(feat_t, states=[h_prev])
        return agents.belief_head(out), new_states[0]
    return belief_step

def ego_feats(state_oh, prev_actions, game):
    B, N, A = state_oh.shape[0], game.N, game.n_actions
    pa  = np.eye(A)[prev_actions]                                  # (B, N, A) one-hot
    ego = np.stack([np.roll(pa, -i, axis=1) for i in range(N)], 1) # (B, N, N, A)  slot 0 = "me"
    ego = ego.reshape(B, N, N * A)
    st  = np.repeat(state_oh[:, None, :], N, axis=1)               # (B, N, n_states)
    f   = np.concatenate([st, ego], axis=2)                        # (B, N, feat_dim)
    return f.reshape(B * N, -1).astype(np.float32)                 # (B*N, feat_dim)

def run_episodes(game, agents, belief_step, hidden=128):
    """Run B episodes in parallel. Returns dict of (B, T, ...) arrays."""
    game.reset()
    theta = generate_types(game)
    prev_actions = np.zeros((game.B, game.N), dtype=int)
    h = tf.zeros((game.B * game.N, hidden))
    S_state, S_belief, S_actions, S_reward, S_feat = [], [], [], [], []

    for t in range(game.ep_len):
        obs_all = game._obs()
        state_oh = obs_all[:, 0, :]

        # Compute belief based on observations
        f = ego_feats(state_oh, prev_actions, game)
        b_tf, h = belief_step(tf.constant(f), h)
        h_np = h.numpy()

        # cooperative actions: policy conditions on h (amortized belief)
        state_rep = np.repeat(state_oh, game.N, axis=0)   # (B*N, n_states)
        c_logits  = agents.policy([state_rep, h_np]).numpy()      # (B*N, A)
        a_coop    = choose_action(c_logits).reshape(game.B, game.N)
        
        # --- adversary actions: one batched call, one batched sample ---
        a_logits = agents.adversary(obs_all.reshape(game.B*game.N, game.n_states)).numpy()  # (B*N, A)
        a_adv    = choose_action(a_logits).reshape(game.B, game.N)                        # (B, N)

        # --- apply corruption ---
        actions = np.where(theta == 1, a_adv, a_coop)

        _, r, done = game.step(actions)
        S_state.append(state_oh); S_belief.append(b_tf.numpy().reshape(game.B, game.N, game.N))
        S_actions.append(actions.copy()); S_reward.append(r); S_feat.append(f)
        prev_actions = actions
        if done: break

    stack = lambda L: np.stack(L, axis=1)
    return dict(
        state=stack(S_state), belief=stack(S_belief),
        actions=stack(S_actions), reward=stack(S_reward), feat=stack(S_feat),
        theta=np.repeat(theta[:, None, :], len(S_reward), axis=1),
    )


def random_rollout(game, B):
    """B episodes, all agents act randomly. Returns (mean per-step reward per env (B,), theta (B,N))."""
    theta = generate_types(game, B)          # (B, N)
    total = np.zeros(B, dtype=np.float32)
    for _ in range(game.ep_len):
        actions = np.random.randint(game.n_actions, size=(B, game.N))   # (B, N)
        _, r, done = game.step(actions)            # r: (B,)
        total += r
        if done:
            break
    return total / game.ep_len, theta