import numpy as np
import tensorflow as tf
from core.utils import generate_types, choose_action
from scipy.special import logsumexp

def take_belief_step(agents, belief_module):
    """Build the compiled single-step belief recurrence bound to these agents'
    weights, for whichever belief_module (factored_sigmoid / categorical_eirmappo)
    is active."""
    return belief_module.make_belief_step(agents.belief_cell, agents.belief_head)

def ego_feats(state_oh, prev_actions, game):
    B, N, A = state_oh.shape[0], game.N, game.n_actions
    pa  = np.eye(A)[prev_actions]                                  # (B, N, A) one-hot
    ego = np.stack([np.roll(pa, -i, axis=1) for i in range(N)], 1) # (B, N, N, A)  slot 0 = "me"
    ego = ego.reshape(B, N, N * A)
    st  = np.repeat(state_oh[:, None, :], N, axis=1)               # (B, N, n_states)
    f   = np.concatenate([st, ego], axis=2)                        # (B, N, feat_dim)
    return f.reshape(B * N, -1).astype(np.float32)                 # (B*N, feat_dim)

def belief_feats(state_oh, prev_actions, game):
    B, N, A = state_oh.shape[0], game.N, game.n_actions
    pa  = np.eye(A)[prev_actions].reshape(B, N*A)          # everyone's prev actions
    pa  = np.repeat(pa[:, None, :], N, axis=1)             # (B, N, N*A)
    st  = np.repeat(state_oh[:, None, :], N, axis=1)       # (B, N, n_states)
    f   = np.concatenate([st, pa], axis=2)
    return f.reshape(B*N, -1).astype(np.float32)
    

def run_episodes_noppo(game, agents, belief_step, hidden=128):
    """Run B episodes in parallel. Returns dict of (B, T, ...) arrays."""
    game.reset()
    theta = generate_types(game)
    prev_actions = np.zeros((game.B, game.N), dtype=int)
    h = tf.zeros((game.B * game.N, hidden))
    S_state, S_belief, S_actions, S_reward, S_feat = [], [], [], [], []

    for t in range(game.ep_len):
        obs_all = game._obs()
        state_oh = obs_all[:, 0, :]     # (B, n_states)  all agents see the same state

        # Compute belief based on observations
        f = belief_feats(state_oh, prev_actions, game)
        policy_input, h, b_probs = belief_step(tf.constant(f), h)
        policy_input_np = policy_input.numpy()

        # cooperative actions: policy conditions on the belief representation
        # (raw hidden state for factored belief, softmax probs for categorical belief)
        state_rep = np.repeat(state_oh, game.N, axis=0)   # (B*N, n_states)
        c_logits  = agents.policy([state_rep, policy_input_np]).numpy()      # (B*N, A)
        a_coop    = choose_action(c_logits).reshape(game.B, game.N)

        # --- adversary actions: one batched call, one batched sample ---
        a_logits = agents.adversary(obs_all.reshape(game.B*game.N, game.n_states)).numpy()  # (B*N, A)
        a_adv    = choose_action(a_logits).reshape(game.B, game.N)                        # (B, N)

        # --- apply corruption ---
        actions = np.where(theta == 1, a_adv, a_coop)
        _, r, done = game.step(actions)
        
        S_state.append(state_oh); S_belief.append(b_probs.numpy().reshape(game.B, game.N, -1))
        S_actions.append(actions.copy()); S_reward.append(r); S_feat.append(f)
        prev_actions = actions
        if done: break

    stack = lambda L: np.stack(L, axis=1)
    return dict(
        state=stack(S_state), belief=stack(S_belief),
        actions=stack(S_actions), reward=stack(S_reward), feat=stack(S_feat),
        theta=np.repeat(theta[:, None, :], len(S_reward), axis=1),
    )


def run_episodes(game, agents, belief_step, hidden=128):
    """Run B episodes in parallel. Returns dict of (B, T, ...) arrays."""
    game.reset()
    theta = generate_types(game)
    prev_actions = np.zeros((game.B, game.N), dtype=int)
    h = tf.zeros((game.B * game.N, hidden))
    S_state, S_belief, S_actions, S_reward, S_feat = [], [], [], [], []
    S_old_logp_coop, S_old_logp_adv = [], []            # NEW

    for t in range(game.ep_len):
        obs_all = game._obs()
        state_oh = obs_all[:, 0, :]

        f = belief_feats(state_oh, prev_actions, game)
        policy_input, h, b_probs = belief_step(tf.constant(f), h)
        policy_input_np = policy_input.numpy()

        # cooperative actions + old log-probs
        state_rep = np.repeat(state_oh, game.N, axis=0)
        c_logits  = agents.policy([state_rep, policy_input_np]).numpy()      # (B*N, A)
        a_coop    = choose_action(c_logits).reshape(game.B, game.N)
        c_logp    = c_logits - logsumexp(c_logits, axis=-1, keepdims=True)   # log-softmax (B*N, A)
        old_lp_coop = c_logp[np.arange(game.B*game.N),
                             a_coop.reshape(-1)].reshape(game.B, game.N)     # NEW (B, N)

        # adversary actions + old log-probs
        a_logits = agents.adversary(obs_all.reshape(game.B*game.N, game.n_states)).numpy()
        a_adv    = choose_action(a_logits).reshape(game.B, game.N)
        a_logp   = a_logits - logsumexp(a_logits, axis=-1, keepdims=True)    # log-softmax
        old_lp_adv = a_logp[np.arange(game.B*game.N),
                            a_adv.reshape(-1)].reshape(game.B, game.N)       # NEW (B, N)

        # apply corruption
        actions = np.where(theta == 1, a_adv, a_coop)
        _, r, done = game.step(actions)

        S_state.append(state_oh); S_belief.append(b_probs.numpy().reshape(game.B, game.N, -1))
        S_actions.append(actions.copy()); S_reward.append(r); S_feat.append(f)
        S_old_logp_coop.append(old_lp_coop); S_old_logp_adv.append(old_lp_adv)   # NEW
        prev_actions = actions
        if done: break

    stack = lambda L: np.stack(L, axis=1)
    return dict(
        state=stack(S_state), belief=stack(S_belief),
        actions=stack(S_actions), reward=stack(S_reward), feat=stack(S_feat),
        old_logp_coop=stack(S_old_logp_coop),    # NEW (B, T, N)
        old_logp_adv=stack(S_old_logp_adv),      # NEW (B, T, N)
        theta=np.repeat(theta[:, None, :], len(S_reward), axis=1),
    )


def random_rollout(game, B):
    """B episodes, all agents act randomly. Returns (mean per-step reward per env (B,), theta (B,N))."""
    game.reset()
    theta = generate_types(game)             # (B, N)
    total = np.zeros(B, dtype=np.float32)
    for _ in range(game.ep_len):
        actions = np.random.randint(game.n_actions, size=(B, game.N))   # (B, N)
        _, r, done = game.step(actions)            # r: (B,)
        total += r
        if done:
            break
    return total / game.ep_len, theta
