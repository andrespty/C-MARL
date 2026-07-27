from tensorflow.keras import layers, Model
import numpy as np
import tensorflow as tf

def build_belief(game, hidden=128):
    """Factored belief network.
       Processes a sequence of per-agent observations of OTHERS' behavior and
       outputs, for EACH agent, an independent P(corrupted) via sigmoid.

       Input: sequence of feature vectors, shape (T, feat_dim).
       Output: sequence of belief vectors, shape (T, N), each in (0,1).

       Factored = N independent sigmoids (not a softmax over subsets),
       so it already scales to k>1 with no change."""
    feat_dim = game.n_states + game.N * game.n_actions
    feat_in = layers.Input(shape=(None, feat_dim), name="belief_feats")  # (T, feat_dim)
    h = layers.GRU(hidden, return_sequences=True, name="GRU")(feat_in)
    # N independent sigmoids -> b_j = P(agent j corrupted)
    b = layers.Dense(game.N, activation="sigmoid", name="belief_out")(h)
    return Model(feat_in, b, name="belief")


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

def update_belief(traj, agents, game):
    B, T = traj['reward'].shape
    N = game.N
    feats_seq = traj['feat'].astype(np.float32)                              # already (B*N, T, F)
    # egocentric labels: agent i sees theta rolled by -i (slot 0 = self)
    theta  = traj['theta']                                                    # (B,T,N)
    th_ego = np.stack([np.roll(theta, -i, axis=2) for i in range(N)], axis=1) # (B,N,T,N)
    labels = th_ego.reshape(B*N, T, N).astype(np.float32)                     # (B*N,T,N)
    assert feats_seq.shape[0] == B*N, feats_seq.shape                         # ordering guard
    loss = _belief_train_step(tf.constant(feats_seq), tf.constant(labels),
                              agents.belief, agents.opt_belief)
    return float(loss)

