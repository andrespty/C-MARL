from tensorflow.keras import layers, Model
import numpy as np
import tensorflow as tf


def build_policy(game, hidden, belief_dim):
    """Cooperative policy (shared by all agents).
       Input: [own observation one-hot | belief representation (belief_dim,)] -> action logits.
       belief_dim comes from whichever beliefs/* module is active (see
       beliefs.<module>.POLICY_INPUT_DIM) -- e.g. N+1 for the categorical EIR-MAPPO
       belief, or `hidden` for the factored/amortized belief."""
    obs_in = layers.Input(shape=(game.n_states,), name="observation")
    b_in = layers.Input(shape=(belief_dim,), name="belief_input")
    x = layers.Concatenate()([obs_in, b_in])
    x = layers.Dense(hidden, activation="relu")(x)
    x = layers.Dense(hidden, activation="relu")(x)
    logits = layers.Dense(game.n_actions, name="action_logits")(x)
    return Model([obs_in, b_in], logits, name="policy")


@tf.function(reduce_retracing=True)
def _policy_train_step(feats_seq, state_flat, actions_flat, adv_flat, coop_mask,
                       policy, belief_input_seq_model, opt, belief_dim, train_belief=True):
    with tf.GradientTape() as tape:
        b_seq  = belief_input_seq_model(feats_seq)          # (B*N, T, belief_dim)
        b_flat = tf.reshape(b_seq, (-1, belief_dim))
        logits = policy([state_flat, b_flat])               # (B*N*T, A)
        logp   = tf.nn.log_softmax(logits)
        idx    = tf.stack([tf.range(tf.shape(actions_flat)[0]), actions_flat], axis=1)
        logp_a = tf.gather_nd(logp, idx)
        w      = tf.stop_gradient(coop_mask * adv_flat)
        loss   = -tf.reduce_sum(w * logp_a) / (tf.reduce_sum(coop_mask) + 1e-8)
    vars_ = policy.trainable_variables + (belief_input_seq_model.trainable_variables if train_belief else [])
    grads = tape.gradient(loss, vars_)
    gv    = [(g, v) for g, v in zip(grads, vars_) if g is not None]
    opt.apply_gradients(gv)
    return loss


def update_policy(traj, adv, agents, game, belief_module, hidden=128, train_belief=True):
    """belief_module is one of beliefs.factored_sigmoid / beliefs.categorical_eirmappo
    (whichever build_agents was given) -- it supplies the model that turns a raw
    feature sequence into the exact policy-input sequence for that belief flavor."""
    if not hasattr(agents, "_belief_input_model"):
        agents._belief_input_model = belief_module.policy_input_model(agents.belief)

    B, T = adv.shape
    N = game.N
    belief_dim = belief_module.POLICY_INPUT_DIM(game, hidden)
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
        agents.policy, agents._belief_input_model, agents.opt_policy, belief_dim, train_belief)
    return float(loss)
