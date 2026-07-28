"""Factored belief — this project's own design, NOT the EIR-MAPPO paper's.

N independent sigmoids: b_j = P(agent j corrupted), computed from each agent's
egocentric history (slot 0 = self, excluded from loss/accuracy since an agent
inferring its own type is meaningless). The policy conditions on the RAW GRU
hidden state h (amortized belief), not on the sigmoid probabilities themselves
-- this is what lets it generalize to k>1 adversaries with no architecture
change (see docs/context_handoff.md, Stage 5).

Same module interface as beliefs/categorical_eirmappo.py so algorithms/ and
core/ code can use either one interchangeably:
  build_belief(game, hidden), POLICY_INPUT_DIM(game, hidden),
  get_recurrent_cell(belief_model), get_output_head(belief_model),
  make_belief_step(cell, head), policy_input_model(belief_model),
  update_belief(traj, agents, game)
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model


def build_belief(game, hidden):
    feat_dim = game.n_states + game.N * game.n_actions
    feat_in = layers.Input(shape=(None, feat_dim), name="belief_feats")
    h = layers.GRU(hidden, return_sequences=True, name="GRU")(feat_in)
    b = layers.Dense(game.N, activation="sigmoid", name="belief_out")(h)
    return Model(feat_in, b, name="belief_factored")


def POLICY_INPUT_DIM(game, hidden):
    return hidden


def get_recurrent_cell(belief_model):
    gru = next(l for l in belief_model.layers if isinstance(l, layers.GRU))
    return gru.cell


def get_output_head(belief_model):
    return belief_model.get_layer("belief_out")


def make_belief_step(cell, head):
    """Single-step recurrence for rollout. Returns (policy_input_t, h_next, belief_probs_t):
    policy_input_t is the raw hidden state; belief_probs_t is the sigmoid output
    (kept only for belief-accuracy logging)."""
    @tf.function(reduce_retracing=True)
    def belief_step(feat_t, h_prev):
        out, new_states = cell(feat_t, states=[h_prev])
        h_next = new_states[0]
        b_t = head(out)
        return h_next, h_next, b_t
    return belief_step


def policy_input_model(belief_model):
    """feats_seq (B*N,T,F) -> policy_input_seq (B*N,T,hidden), i.e. the GRU's hidden
    sequence itself (pre-sigmoid-head)."""
    gru = next(l for l in belief_model.layers if isinstance(l, layers.GRU))
    return Model(belief_model.input, gru.output)


@tf.function(reduce_retracing=True)
def _belief_train_step(feats_seq, labels_ego, belief, opt):
    # feats_seq: (B*N, T, feat_dim)   labels_ego: (B*N, T, N)  slot0 = self
    with tf.GradientTape() as tape:
        b_pred = belief(feats_seq)
        bce = tf.keras.losses.binary_crossentropy(labels_ego[..., 1:],  # drop self-slot
                                                  b_pred[..., 1:])
        loss = tf.reduce_mean(bce)
    grads = tape.gradient(loss, belief.trainable_variables)
    opt.apply_gradients(zip(grads, belief.trainable_variables))
    return loss


def update_belief(traj, agents, game):
    B, T = traj['reward'].shape
    N = game.N
    feats_seq = traj['feat'].astype(np.float32)                              # (B*N, T, F)
    # egocentric labels: agent i sees theta rolled by -i (slot 0 = self)
    theta  = traj['theta']                                                    # (B,T,N)
    th_ego = np.stack([np.roll(theta, -i, axis=2) for i in range(N)], axis=1) # (B,N,T,N)
    labels = th_ego.reshape(B * N, T, N).astype(np.float32)                   # (B*N,T,N)
    assert feats_seq.shape[0] == B * N, feats_seq.shape                       # ordering guard
    loss = _belief_train_step(tf.constant(feats_seq), tf.constant(labels),
                              agents.belief, agents.opt_belief)
    return float(loss)


def belief_accuracy(ev, game):
    """ev = run_episodes(...) output. Thresholded per-agent sigmoid accuracy,
    dropping the meaningless self-slot."""
    th = ev['theta']                                                          # (B,T,N)
    th_ego = np.stack([np.roll(th, -i, axis=2) for i in range(game.N)], axis=2)  # (B,T,N,N)
    pred = (ev['belief'] > 0.5).astype(int)
    return float((pred[..., 1:] == th_ego[..., 1:]).mean())
