from tensorflow.keras import layers, Model
import numpy as np
import tensorflow as tf

def build_value_net(game, hidden=128):
    """Centralized critic (training only).
       Input: [global state one-hot | belief vector b (N,)] -> scalar value."""
    state_in = layers.Input(shape=(game.n_states,), name="state")
    belief_in = layers.Input(shape=(game.N,), name="belief")
    x = layers.Concatenate()([state_in, belief_in])
    x = layers.Dense(hidden, activation="relu")(x)
    x = layers.Dense(hidden, activation="relu")(x)
    value = layers.Dense(1, name="value")(x)
    return Model([state_in, belief_in], value, name="value_net")

@tf.function(reduce_retracing=True)
def _critic_train_step(state, belief_in, target, critic, opt):
    with tf.GradientTape() as tape:
        V = critic([state, belief_in])
        loss = tf.reduce_mean((target - V) ** 2)
    grads = tape.gradient(loss, critic.trainable_variables)
    opt.apply_gradients(zip(grads, critic.trainable_variables))
    return loss

def update_value_net(traj, G, agents, game):
    B, T = G.shape
    state  = traj['state'].reshape(B*T, game.n_states).astype(np.float32)
    theta  = traj['theta'].reshape(B*T, game.N).astype(np.float32)   # true types, not belief
    target = G.reshape(B*T, 1).astype(np.float32)
    loss = _critic_train_step(tf.constant(state), tf.constant(theta),
                              tf.constant(target), agents.value, agents.opt_value)
    return loss

