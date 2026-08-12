"""Call these from the notebook to see a belief module's inputs/outputs.
Pass belief_module=beliefs.factored_sigmoid or beliefs.categorical_eirmappo."""
import numpy as np
import tensorflow as tf
from game_environments.ToyGame import ToyGame

def check_belief(belief_module, game=None, hidden=128, B=4, T=5):
    """Full-sequence belief forward pass, like what update_belief sees."""
    game = game or ToyGame(B)
    belief = belief_module.build_belief(game, hidden)

    feat_dim = game.n_states + game.N * game.n_actions
    feats = np.random.randn(game.N * B, T, feat_dim).astype(np.float32)
    out = belief(feats).numpy()

    print(f"feats   {feats.shape}")
    print(f"belief  {out.shape}   range [{out.min():.3f}, {out.max():.3f}]")
    print(out[0])
    return out

def check_belief_step(belief_module, game=None, hidden=128, B=4):
    """Single-step recurrence, like what run_episodes calls every timestep."""
    game = game or ToyGame(B)
    belief = belief_module.build_belief(game, hidden)
    cell = belief_module.get_recurrent_cell(belief)
    head = belief_module.get_output_head(belief)
    step = belief_module.make_belief_step(cell, head)

    feat_dim = game.n_states + game.N * game.n_actions
    feat_t = tf.constant(np.random.randn(game.N * B, feat_dim).astype(np.float32))
    h_prev = tf.zeros((game.N * B, hidden))
    policy_input, h_next, belief_probs = step(feat_t, h_prev)

    print(f"policy_input  {policy_input.shape}")
    print(f"h_next        {h_next.shape}")
    print(f"belief_probs  {belief_probs.shape}")
    print(belief_probs.numpy()[0])
    return policy_input, h_next, belief_probs

