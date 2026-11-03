import torch
from cmarl.environments.toy import ToyGame
from cmarl.byzantine.priors import SingleAdversary
from cmarl.byzantine.wrapper import ByzantineEnv
from cmarl.agents.belief import CategoricalBelief
from cmarl.agents.policy import TeamPolicy, AdversaryPolicy
from cmarl.agents.critic import VCritic
from cmarl.algorithms.gda import GDA, GDAConfig
from cmarl.evaluate import evaluate_robustness

torch.manual_seed(0)
env = ByzantineEnv(ToyGame(num_envs=256, episode_length=50), SingleAdversary(n_agents=2))
belief = CategoricalBelief(env.spec, env.prior)
algo = GDA(
    team=TeamPolicy(env.spec, belief_dim=belief.policy_dim),
    adversary=AdversaryPolicy(env.spec),
    belief=belief,
    critic=VCritic(env.spec),
    config=GDAConfig(team_interval=10, entropy_coef=0.01),
)
history = algo.train(env, n_iters=1000, log_every=50)


result = evaluate_robustness(env, algo.team, algo.belief, n_iters=1000)
print("robust:", result["robust_reward"], "worst:", result["worst_reward"])