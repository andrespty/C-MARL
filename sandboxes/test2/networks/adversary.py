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
def _adversary_train_step(state_rep, actions_flat, weight, norm, adversary, opt):
    with tf.GradientTape() as tape:
        logits = adversary(state_rep)
        logp = tf.nn.log_softmax(logits, axis=-1)
        lp = tf.gather(logp, actions_flat, batch_dims=1)
        loss = tf.reduce_sum(tf.stop_gradient(weight) * lp) / norm   # + sign: adversary minimizes team return
    grads = tape.gradient(loss, adversary.trainable_variables)
    return loss, grads

def update_adversary(traj, adv, agents, game):
    """Policy gradient, adversary only."""
    B, T = adv.shape
    state    = traj['state'].reshape(B*T, game.n_states)
    actions  = traj['actions'].reshape(B*T, game.N)
    theta    = traj['theta'].reshape(B*T, game.N).astype(np.float32)
    adv_flat = adv.reshape(B*T)

    state_rep    = np.repeat(state, game.N, axis=0).astype(np.float32)
    actions_flat = actions.reshape(B*T*game.N).astype(np.int32)
    weight       = (theta * adv_flat[:, None]).reshape(B*T*game.N).astype(np.float32)
    norm         = np.float32(np.sum(theta) + 1e-8)

    loss, grads = _adversary_train_step(
        tf.constant(state_rep), tf.constant(actions_flat),
        tf.constant(weight), tf.constant(norm),
        agents.adversary, agents.opt_adv
    )
    agents.opt_adv.apply_gradients(zip(grads, agents.adversary.trainable_variables))
    return float(loss)
