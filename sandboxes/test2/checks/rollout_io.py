"""Call this from the notebook to see the shape of everything run_episodes
returns, for a given belief module."""
from game_environments.ToyGame import ToyGame
from core.agents import build_agents
from core.rollout import run_episodes, take_belief_step


def check_rollout(belief_module, game=None, hidden=128, B=4):
    game = game or ToyGame(B)
    agents = build_agents(game, belief_module, hidden=hidden)
    belief_step = take_belief_step(agents, belief_module)
    traj = run_episodes(game, agents, belief_step, hidden=hidden)

    for key, val in traj.items():
        print(f"{key:10s} {val.shape}")
    return traj
