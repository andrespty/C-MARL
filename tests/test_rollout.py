from cmarl.environments.toy import ToyGame
from cmarl.byzantine.priors import SingleAdversary
from cmarl.byzantine.wrapper import ByzantineEnv
from cmarl.agents.belief import CategoricalBelief
from cmarl.agents.policy import TeamPolicy, AdversaryPolicy
from cmarl.rollout import collect

env = ByzantineEnv(ToyGame(num_envs=8, episode_length=1000), SingleAdversary(n_agents=2))
belief = CategoricalBelief(env.spec, env.prior)
team = TeamPolicy(env.spec, belief_dim=belief.policy_dim)
adv = AdversaryPolicy(env.spec)

batch = collect(env, team, adv, belief)
for name in ["obs", "state", "feat", "actions", "logp", "reward", "theta"]:
    print(f"{name:8s}", tuple(getattr(batch, name).shape))
print("mean reward:", batch.reward.mean().item())      # ≈ 0.5 with random policies