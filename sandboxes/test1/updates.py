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
    state_flat  = traj['state'].reshape(B*T, cfg.n_states)
    belief_flat = traj['belief'].reshape(B*T, cfg.N)
    V = agents.critic([state_flat, belief_flat]).numpy().reshape(B, T)
    adv = G - V
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return G.astype(np.float32), adv.astype(np.float32)

@tf.function(reduce_retracing=True)
def _critic_train_step(state, belief_in, target, critic, opt):
    with tf.GradientTape() as tape:
        V = critic([state, belief_in])
        loss = tf.reduce_mean((target - V) ** 2)
    grads = tape.gradient(loss, critic.trainable_variables)
    opt.apply_gradients(zip(grads, critic.trainable_variables))
    return loss

def update_critic(traj, G, agents, cfg):
    B, T = G.shape
    state     = traj['state'].reshape(B*T, cfg.n_states).astype(np.float32)
    belief_in = traj['belief'].reshape(B*T, cfg.N).astype(np.float32)
    target    = G.reshape(B*T, 1).astype(np.float32)
    loss = _critic_train_step(tf.constant(state), tf.constant(belief_in),
                              tf.constant(target), agents.critic, agents.opt_critic)
    return float(loss)


@tf.function(reduce_retracing=True)
def _belief_train_step(feats, labels, belief, opt):
    with tf.GradientTape() as tape:
        b_pred = belief(feats)
        loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(labels, b_pred))
    grads = tape.gradient(loss, belief.trainable_variables)
    opt.apply_gradients(zip(grads, belief.trainable_variables))
    return loss

def update_belief(traj, agents, cfg):
    feats  = traj['feat']
    labels = traj['theta'].astype(np.float32)
    loss = _belief_train_step(
        tf.constant(feats), tf.constant(labels),
        agents.belief, agents.opt_belief
    )
    return float(loss)

@tf.function(reduce_retracing=True)
def _policy_train_step(state_rep, b_ego_flat, actions_flat, weight, norm, policy, opt):
    with tf.GradientTape() as tape:
        logits = policy([state_rep, b_ego_flat])
        logp = tf.nn.log_softmax(logits, axis=-1)
        lp = tf.gather(logp, actions_flat, batch_dims=1)
        loss = -tf.reduce_sum(tf.stop_gradient(weight) * lp) / norm
    grads = tape.gradient(loss, policy.trainable_variables)
    opt.apply_gradients(zip(grads, policy.trainable_variables))
    return loss

def update_policy(traj, adv, agents, cfg):
    """Policy gradient, cooperative agents only."""
    B, T = adv.shape
    state     = traj['state'].reshape(B*T, cfg.n_states)
    belief_in = traj['belief'].reshape(B*T, cfg.N)
    actions   = traj['actions'].reshape(B*T, cfg.N)
    theta     = traj['theta'].reshape(B*T, cfg.N)
    adv_flat  = adv.reshape(B*T)
    coop_mask = (1 - theta).astype(np.float32)

    b_ego = np.stack([np.roll(belief_in, -i, axis=1) for i in range(cfg.N)], axis=1)
    b_ego_flat   = b_ego.reshape(B*T*cfg.N, cfg.N).astype(np.float32)
    state_rep    = np.repeat(state, cfg.N, axis=0).astype(np.float32)
    actions_flat = actions.reshape(B*T*cfg.N).astype(np.int32)
    weight       = (coop_mask * adv_flat[:, None]).reshape(B*T*cfg.N).astype(np.float32)
    norm         = np.float32(np.sum(coop_mask) + 1e-8)

    loss = _policy_train_step(
        tf.constant(state_rep), tf.constant(b_ego_flat),
        tf.constant(actions_flat), tf.constant(weight), tf.constant(norm),
        agents.policy, agents.opt_policy
    )
    return float(loss)


@tf.function(reduce_retracing=True)
def _adversary_train_step(state_rep, actions_flat, weight, norm, adversary, opt):
    with tf.GradientTape() as tape:
        logits = adversary(state_rep)
        logp = tf.nn.log_softmax(logits, axis=-1)
        lp = tf.gather(logp, actions_flat, batch_dims=1)
        loss = tf.reduce_sum(tf.stop_gradient(weight) * lp) / norm   # + sign: adversary minimizes team return
    grads = tape.gradient(loss, adversary.trainable_variables)
    opt.apply_gradients(zip(grads, adversary.trainable_variables))
    return loss

def update_adversary(traj, adv, agents, cfg):
    """Policy gradient, adversary only."""
    B, T = adv.shape
    state    = traj['state'].reshape(B*T, cfg.n_states)
    actions  = traj['actions'].reshape(B*T, cfg.N)
    theta    = traj['theta'].reshape(B*T, cfg.N).astype(np.float32)
    adv_flat = adv.reshape(B*T)

    state_rep    = np.repeat(state, cfg.N, axis=0).astype(np.float32)
    actions_flat = actions.reshape(B*T*cfg.N).astype(np.int32)
    weight       = (theta * adv_flat[:, None]).reshape(B*T*cfg.N).astype(np.float32)
    norm         = np.float32(np.sum(theta) + 1e-8)

    loss = _adversary_train_step(
        tf.constant(state_rep), tf.constant(actions_flat),
        tf.constant(weight), tf.constant(norm),
        agents.adversary, agents.opt_adv
    )
    return float(loss)