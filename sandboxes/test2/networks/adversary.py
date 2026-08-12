from tensorflow.keras import layers, Model
import numpy as np
import tensorflow as tf

def build_adversary(game, hidden=128):
    """Adversary policy (shared; used for whichever agent is corrupted).
       Input: own observation one-hot -> action logits. (knows it's the adversary)"""
    obs_in = layers.Input(shape=(game.n_states,), name="observation")
    x = layers.Dense(hidden, activation="relu")(obs_in)
    x = layers.Dense(hidden, activation="relu")(x)
    logits = layers.Dense(game.n_actions, name="adv_logits")(x)
    return Model(obs_in, logits, name="adversary")

@tf.function(reduce_retracing=True)
def _adversary_train_step(state_rep, actions_flat, weight, old_logp, norm,
                          adversary, clip=0.2):
    with tf.GradientTape() as tape:
        logits = adversary(state_rep)
        logp = tf.nn.log_softmax(logits, axis=-1)
        new_logp = tf.gather(logp, actions_flat, batch_dims=1)      # (M,)

        ratio     = tf.exp(new_logp - old_logp)                     # π_new / π_old
        w         = tf.stop_gradient(weight)
        unclipped = ratio * w
        clipped   = tf.clip_by_value(ratio, 1.0 - clip, 1.0 + clip) * w
        # + sign: adversary MINIMIZES team return.
        # PPO takes the pessimistic bound for the maximizer; the adversary is a
        # minimizer, so we mirror it: maximize over the clipped range -> use tf.maximum.
        loss = tf.reduce_sum(tf.maximum(unclipped, clipped)) / norm
    grads = tape.gradient(loss, adversary.trainable_variables)
    return loss, grads

def update_adversary(traj, adv, agents, game, clip=0.2):
    """PPO-clipped policy gradient, adversary only."""
    B, T = adv.shape
    state    = traj['state'].reshape(B*T, game.n_states)
    actions  = traj['actions'].reshape(B*T, game.N)
    theta    = traj['theta'].reshape(B*T, game.N).astype(np.float32)
    adv_flat = adv.reshape(B*T)

    state_rep    = np.repeat(state, game.N, axis=0).astype(np.float32)
    actions_flat = actions.reshape(B*T*game.N).astype(np.int32)
    weight       = (theta * adv_flat[:, None]).reshape(B*T*game.N).astype(np.float32)
    norm         = np.float32(np.sum(theta) + 1e-8)
    # old_logp flattened to match actions_flat: (B, T, N) -> (B*T, N) -> (B*T*N,)
    old_logp     = traj['old_logp_adv'].reshape(B*T, game.N).reshape(B*T*game.N).astype(np.float32)

    loss, grads = _adversary_train_step(
        tf.constant(state_rep), tf.constant(actions_flat),
        tf.constant(weight), tf.constant(old_logp), tf.constant(norm),
        agents.adversary, clip
    )
    agents.opt_adv.apply_gradients(zip(grads, agents.adversary.trainable_variables))
    return float(loss)