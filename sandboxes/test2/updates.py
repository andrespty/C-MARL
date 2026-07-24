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
    V = agents.critic([state_flat, theta_flat]).numpy().reshape(B, T)
    adv = G - V
    return G, adv

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
    state  = traj['state'].reshape(B*T, cfg.n_states).astype(np.float32)
    theta  = traj['theta'].reshape(B*T, cfg.N).astype(np.float32)   # true types, not belief
    target = G.reshape(B*T, 1).astype(np.float32)
    loss = _critic_train_step(tf.constant(state), tf.constant(theta),
                              tf.constant(target), agents.critic, agents.opt_critic)
    return loss

@tf.function(reduce_retracing=True)
def _belief_train_step(feats_seq, labels_ego, belief, opt):
    # feats_seq: (B*N, T, feat_dim)   labels_ego: (B*N, T, N)  slot0 = self
    with tf.GradientTape() as tape:
        out = belief(feats_seq)
        b_pred = out[-1] if isinstance(out, (list, tuple)) else out     # (B*N, T, N)
        bce = tf.keras.losses.binary_crossentropy(labels_ego[..., 1:],  # drop self-slot
                                                  b_pred[..., 1:])
        loss = tf.reduce_mean(bce)
    grads = tape.gradient(loss, belief.trainable_variables)
    opt.apply_gradients(zip(grads, belief.trainable_variables))
    return loss

def update_belief(traj, agents, cfg):
    B, T = traj['reward'].shape
    N = cfg.N
    feats_seq = traj['feat'].astype(np.float32)                              # already (B*N, T, F)
    # egocentric labels: agent i sees theta rolled by -i (slot 0 = self)
    theta  = traj['theta']                                                    # (B,T,N)
    th_ego = np.stack([np.roll(theta, -i, axis=2) for i in range(N)], axis=1) # (B,N,T,N)
    labels = th_ego.reshape(B*N, T, N).astype(np.float32)                     # (B*N,T,N)
    assert feats_seq.shape[0] == B*N, feats_seq.shape                         # ordering guard
    loss = _belief_train_step(tf.constant(feats_seq), tf.constant(labels),
                              agents.belief, agents.opt_belief)
    return float(loss)

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
        w      = coop_mask * adv_flat
        loss   = -tf.reduce_sum(w * logp_a) / (tf.reduce_sum(coop_mask) + 1e-8)
    vars_ = policy.trainable_variables + (belief_h.trainable_variables if train_belief else [])
    grads = tape.gradient(loss, vars_)
    gv    = [(g, v) for g, v in zip(grads, vars_) if g is not None]
    opt.apply_gradients(gv)
    return loss

def update_policy(traj, adv, agents, cfg, hidden=128, train_belief=True):
    # build the hidden-state extractor once (shares weights with agents.belief)
    if not hasattr(agents, "belief_h"):
        gru = next(l for l in agents.belief.layers if isinstance(l, tf.keras.layers.GRU))
        agents.belief_h = tf.keras.Model(agents.belief.input, gru.output)  # -> (B*N, T, hidden)

    B, T = adv.shape
    N = cfg.N
    feats_seq  = traj['feat'].astype(np.float32)                                  # (B*N, T, F)
    state      = traj['state']
    state_flat = np.repeat(state[:, None, :, :], N, axis=1).reshape(B*N*T, cfg.n_states).astype(np.float32)
    actions_fl = np.transpose(traj['actions'], (0, 2, 1)).reshape(-1).astype(np.int32)
    theta_fl   = np.transpose(traj['theta'],   (0, 2, 1)).reshape(-1).astype(np.float32)
    coop_mask  = (theta_fl == 0).astype(np.float32)
    adv_flat   = np.repeat(adv[:, None, :], N, axis=1).reshape(-1).astype(np.float32)
    loss = _policy_train_step(
        tf.constant(feats_seq), tf.constant(state_flat),
        tf.constant(actions_fl), tf.constant(adv_flat), tf.constant(coop_mask),
        agents.policy, agents.belief_h, agents.opt_policy, hidden, train_belief)
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