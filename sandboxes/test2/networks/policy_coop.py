from tensorflow.keras import layers, Model
import numpy as np
import tensorflow as tf

def build_policy(game, hidden=128):
    """Cooperative policy (shared by all agents).
       Input: [own observation one-hot | belief vector b (N,)] -> action logits."""
    obs_in = layers.Input(shape=(game.n_states,), name="observation")
    h_in = layers.Input(shape=(hidden,), name="belief_state")
    x = layers.Concatenate()([obs_in, h_in])
    x = layers.Dense(hidden, activation="relu")(x)
    x = layers.Dense(hidden, activation="relu")(x)
    logits = layers.Dense(game.n_actions, name="action_logits")(x)
    return Model([obs_in, h_in], logits, name="policy")

@tf.function(reduce_retracing=True)
def _policy_train_step(feats_seq, state_flat, actions_flat, adv_flat, coop_mask,
                       policy, belief_h, opt, hidden, train_belief=True):
    with tf.GradientTape() as tape:
        h_seq  = belief_h(feats_seq)                      # (B*N, T, hidden)  <-- real hidden state
        h_flat = tf.reshape(h_seq, (-1, hidden))          # (B*N*T, hidden)
        logits = policy([state_flat, h_flat])             # (B*N*T, A)
        logp   = tf.nn.log_softmax(logits)
        idx    = tf.stack([tf.range(tf.shape(actions_flat)[0]), actions_flat], axis=1)
        logp_a = tf.gather_nd(logp, idx)
        w      = tf.stop_gradient(coop_mask * adv_flat)
        loss   = -tf.reduce_sum(w * logp_a) / (tf.reduce_sum(coop_mask) + 1e-8)
    vars_ = policy.trainable_variables + (belief_h.trainable_variables if train_belief else [])
    grads = tape.gradient(loss, vars_)
    gv    = [(g, v) for g, v in zip(grads, vars_) if g is not None]
    opt.apply_gradients(gv)
    return loss

def update_policy(traj, adv, agents, game, hidden=128, train_belief=True):
    # build the hidden-state extractor once (shares weights with agents.belief)
    if not hasattr(agents, "belief_h"):
        gru = next(l for l in agents.belief.layers if isinstance(l, tf.keras.layers.GRU))
        agents.belief_h = tf.keras.Model(agents.belief.input, gru.output)  # -> (B*N, T, hidden)

    B, T = adv.shape
    N = game.N
    feats_seq  = traj['feat'].astype(np.float32)                                  # (B*N, T, F)
    state      = traj['state']
    state_flat = np.repeat(state[:, None, :, :], N, axis=1).reshape(B*N*T, game.n_states).astype(np.float32)
    actions_fl = np.transpose(traj['actions'], (0, 2, 1)).reshape(-1).astype(np.int32)
    theta_fl   = np.transpose(traj['theta'],   (0, 2, 1)).reshape(-1).astype(np.float32)
    coop_mask  = (theta_fl == 0).astype(np.float32)
    adv_flat   = np.repeat(adv[:, None, :], N, axis=1).reshape(-1).astype(np.float32)
    loss = _policy_train_step(
        tf.constant(feats_seq), tf.constant(state_flat),
        tf.constant(actions_fl), tf.constant(adv_flat), tf.constant(coop_mask),
        agents.policy, agents.belief_h, agents.opt_policy, hidden, train_belief)
    return float(loss)