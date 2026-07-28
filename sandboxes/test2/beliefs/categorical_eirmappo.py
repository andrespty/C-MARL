"""Faithful EIR-MAPPO belief (Li et al. 2024, ICLR).

(N+1)-way categorical distribution over "which single agent, if any, is the
adversary": class 0 = no adversary, class k (1..N) = the agent in ego-rolled
position k-1 is the adversary (rolled so position 0 = self -- class 1 means
"I am flagged"). Matches the paper's model of exactly one adversary per
episode. The belief probability vector itself (not a raw hidden state) is
what gets fed to the policy -- this is the key difference from
beliefs/factored_sigmoid.py, which was this project's own later modification
to support k>1 adversaries.

ASSUMPTION TO VERIFY AGAINST THE PAPER: the egocentric roll-per-agent belief
call (reusing this project's ego_feats plumbing) is what lets the shared-weight
policy specialize per agent -- without it, both agents would receive an
identical belief vector each timestep and couldn't tell "me" from "partner"
(this was the exact bug documented as the "egocentric belief fix" in
docs/context_handoff.md). If the paper instead computes a single *global*
belief shared verbatim across agents, this module's roll-and-relabel scheme
would need to be dropped in favor of that -- flag this back if it doesn't
match your reading of the paper.

Same module interface as beliefs/factored_sigmoid.py.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model


def build_belief(game, hidden):
    feat_dim = game.n_states + game.N * game.n_actions
    feat_in = layers.Input(shape=(None, feat_dim), name="belief_feats")
    h = layers.GRU(hidden, return_sequences=True, name="GRU")(feat_in)
    b = layers.Dense(game.N + 1, activation="softmax", name="belief_out")(h)
    return Model(feat_in, b, name="belief_categorical")


def POLICY_INPUT_DIM(game, hidden):
    return game.N + 1


def get_recurrent_cell(belief_model):
    gru = next(l for l in belief_model.layers if isinstance(l, layers.GRU))
    return gru.cell


def get_output_head(belief_model):
    return belief_model.get_layer("belief_out")


def make_belief_step(cell, head):
    """Single-step recurrence for rollout. Returns (policy_input_t, h_next, belief_probs_t):
    here policy_input_t IS belief_probs_t (the softmax itself) -- h_next is only
    threaded forward internally, never seen by the policy."""
    @tf.function(reduce_retracing=True)
    def belief_step(feat_t, h_prev):
        out, new_states = cell(feat_t, states=[h_prev])
        b_t = head(out)
        return b_t, new_states[0], b_t
    return belief_step


def policy_input_model(belief_model):
    """feats_seq (B*N,T,F) -> policy_input_seq (B*N,T,N+1): the belief model's
    own output sequence already IS the policy input, no need to reach inside."""
    return Model(belief_model.input, belief_model.output)


def _ego_class_labels(theta, N):
    """theta: (B,T,N) true types -> (B,N,T) int class labels, ego-rolled per agent i.
    0 = no adversary this episode, k = agent at rolled position k-1 is the adversary."""
    th_ego = np.stack([np.roll(theta, -i, axis=2) for i in range(N)], axis=1)  # (B,N,T,N)
    has_adv = th_ego.sum(axis=-1) > 0
    labels = np.where(has_adv, th_ego.argmax(axis=-1) + 1, 0)
    return labels.astype(np.int32)  # (B,N,T)


@tf.function(reduce_retracing=True)
def _belief_train_step(feats_seq, labels, belief, opt):
    with tf.GradientTape() as tape:
        b_pred = belief(feats_seq)                          # (B*N, T, N+1)
        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(labels, b_pred)
        )
    grads = tape.gradient(loss, belief.trainable_variables)
    opt.apply_gradients(zip(grads, belief.trainable_variables))
    return loss


def update_belief(traj, agents, game):
    B, T = traj['reward'].shape
    N = game.N
    feats_seq = traj['feat'].astype(np.float32)                     # (B*N, T, F)
    labels = _ego_class_labels(traj['theta'], N).reshape(B * N, T)  # (B*N, T)
    assert feats_seq.shape[0] == B * N, feats_seq.shape
    loss = _belief_train_step(tf.constant(feats_seq), tf.constant(labels),
                              agents.belief, agents.opt_belief)
    return float(loss)


def belief_accuracy(ev, game):
    """ev = run_episodes(...) output. Argmax-class accuracy against the true
    ego-rolled (N+1)-way label (0 = no adversary, k = agent at rolled slot k-1)."""
    th = ev['theta']                                                             # (B,T,N)
    th_ego = np.stack([np.roll(th, -i, axis=2) for i in range(game.N)], axis=2)   # (B,T,N,N)
    has_adv = th_ego.sum(axis=-1) > 0
    labels = np.where(has_adv, th_ego.argmax(axis=-1) + 1, 0)                     # (B,T,N)
    pred = ev['belief'].argmax(axis=-1)                                          # (B,T,N)
    return float((pred == labels).mean())
