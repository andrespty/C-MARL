from tensorflow.keras import layers, Model
import numpy as np

def build_policy(cfg, hidden=128):
    """Cooperative policy (shared by all agents).
       Input: [own observation one-hot | belief vector b (N,)] -> action logits."""
    obs_in = layers.Input(shape=(cfg.n_states,), name="observation")
    h_in = layers.Input(shape=(hidden,), name="belief_state")
    x = layers.Concatenate()([obs_in, h_in])
    x = layers.Dense(hidden, activation="relu")(x)
    x = layers.Dense(hidden, activation="relu")(x)
    logits = layers.Dense(cfg.n_actions, name="action_logits")(x)
    return Model([obs_in, h_in], logits, name="policy")

def build_adversary(cfg, hidden=128):
    """Adversary policy (shared; used for whichever agent is corrupted).
       Input: own observation one-hot -> action logits. (knows it's the adversary)"""
    obs_in = layers.Input(shape=(cfg.n_states,), name="observation")
    x = layers.Dense(hidden, activation="relu")(obs_in)
    x = layers.Dense(hidden, activation="relu")(x)
    logits = layers.Dense(cfg.n_actions, name="adv_logits")(x)
    return Model(obs_in, logits, name="adversary")

def build_critic(cfg, hidden=128):
    """Centralized critic (training only).
       Input: [global state one-hot | belief vector b (N,)] -> scalar value."""
    state_in = layers.Input(shape=(cfg.n_states,), name="state")
    belief_in = layers.Input(shape=(cfg.N,), name="belief")
    x = layers.Concatenate()([state_in, belief_in])
    x = layers.Dense(hidden, activation="relu")(x)
    x = layers.Dense(hidden, activation="relu")(x)
    value = layers.Dense(1, name="value")(x)
    return Model([state_in, belief_in], value, name="critic")

def build_belief(cfg, hidden=128):
    """Factored belief network.
       Processes a sequence of per-agent observations of OTHERS' behavior and
       outputs, for EACH agent, an independent P(corrupted) via sigmoid.

       Input: sequence of feature vectors, shape (T, feat_dim).
       Output: sequence of belief vectors, shape (T, N), each in (0,1).

       Factored = N independent sigmoids (not a softmax over subsets),
       so it already scales to k>1 with no change."""
    feat_dim = cfg.n_states + cfg.N * cfg.n_actions
    feat_in = layers.Input(shape=(None, feat_dim), name="belief_feats")  # (T, feat_dim)
    h = layers.GRU(hidden, return_sequences=True, name="GRU")(feat_in)
    # N independent sigmoids -> b_j = P(agent j corrupted)
    b = layers.Dense(cfg.N, activation="sigmoid", name="belief_out")(h)
    return Model(feat_in, b, name="belief")
