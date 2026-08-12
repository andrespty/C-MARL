"""Call these from the notebook to see what a network's inputs/outputs
actually look like. Each one builds a fresh network, feeds it a small random
batch, prints the shapes, and returns the output so you can keep poking at it."""
import numpy as np

from game_environments.ToyGame import ToyGame
from networks.policy import build_policy
from networks.adversary import build_adversary
from networks.value_net import build_value_net


def check_policy(game=None, hidden=128, belief_dim=None, B=4):
    game = game or ToyGame(B)
    belief_dim = belief_dim or hidden
    policy = build_policy(game, hidden, belief_dim)

    obs = np.eye(game.n_states, dtype=np.float32)[np.random.randint(game.n_states, size=B)]
    belief_input = np.random.randn(B, belief_dim).astype(np.float32)
    logits = policy([obs, belief_input]).numpy()

    print(f"obs           {obs.shape}")
    print(f"belief_input  {belief_input.shape}")
    print(f"logits        {logits.shape}")
    print(logits)
    return logits


def check_adversary(game=None, hidden=128, B=4):
    game = game or ToyGame(B)
    adversary = build_adversary(game, hidden)

    obs = np.eye(game.n_states, dtype=np.float32)[np.random.randint(game.n_states, size=B)]
    logits = adversary(obs).numpy()

    print(f"obs     {obs.shape}")
    print(f"logits  {logits.shape}")
    print(logits)
    return logits


def check_value_net(game=None, hidden=128, B=4):
    game = game or ToyGame(B)
    value = build_value_net(game, hidden)

    state = np.eye(game.n_states, dtype=np.float32)[np.random.randint(game.n_states, size=B)]
    theta = (np.random.rand(B, game.N) < 0.5).astype(np.float32)
    v = value([state, theta]).numpy()

    print(f"state  {state.shape}")
    print(f"theta  {theta.shape}")
    print(f"value  {v.shape}")
    print(v)
    return v
