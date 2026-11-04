import torch
from cmarl.environments.toy import ToyGame
from cmarl.byzantine.priors import AdaptiveSingleAdversary, SingleAdversary
from cmarl.byzantine.wrapper import ByzantineEnv
from cmarl.agents.belief import CategoricalBelief
from cmarl.agents.policy import TeamPolicy, AdversaryPolicy
from cmarl.agents.critic import VCritic
from cmarl.algorithms.gda import GDA, GDAConfig
from cmarl.evaluate import evaluate_robustness

torch.manual_seed(0)
T = 100
H = (128, 128, 128)
INTERVAL = 10

env = ByzantineEnv(
    ToyGame(num_envs=40, episode_length=T),
    AdaptiveSingleAdversary(n_agents=2, temperature=1 / T, interval=INTERVAL),
)
belief = CategoricalBelief(env.spec, env.prior)
config = GDAConfig(
    clip=0.05,
    lr_team=5e-4, lr_adv=5e-4, lr_critic=5e-4, lr_belief=5e-4,
    max_grad_norm=10.0,
    entropy_coef=0.01,
    team_interval=INTERVAL,
)
print(config)

algo = GDA(
    team=TeamPolicy(env.spec, belief_dim=belief.policy_dim, hidden_sizes=H),
    adversary=AdversaryPolicy(env.spec, hidden_sizes=H),
    belief=belief,
    critic=VCritic(env.spec, belief_dim=belief.policy_dim, hidden_sizes=H),   # FIXED
    config=config,
)
history = algo.train(env, n_iters=10_000, log_every=100)

# Separate env for evaluation: plain prior, every agent attacked equally often.
eval_env = ByzantineEnv(ToyGame(num_envs=40, episode_length=T), SingleAdversary(n_agents=2))   # FIXED
result = evaluate_robustness(eval_env, algo.team, algo.belief, n_iters=1000,
                             config=config, hidden_sizes=H)                                    # FIXED
print("robust:", result["robust_reward"], "worst:", result["worst_reward"])