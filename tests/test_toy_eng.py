import torch
from cmarl.environments.toy import ToyGame
from cmarl.byzantine.priors import SingleAdversary
from cmarl.byzantine.wrapper import ByzantineEnv
from cmarl.agents.belief import CategoricalBelief
from cmarl.agents.policy import TeamPolicy, AdversaryPolicy
from cmarl.agents.critic import QCritic
from cmarl.algorithms.base import reward_stats
from cmarl.algorithms.energy import Energy, EnergyConfig
from cmarl.algorithms.gda import GDAConfig
from cmarl.evaluate import analytical_robustness, evaluate_robustness
from cmarl.rollout import collect

T = 100
H = (128, 128, 128)
N_ITERS = 2_000
SEED = 0

torch.manual_seed(SEED)
env = ByzantineEnv(ToyGame(num_envs=40, episode_length=T), SingleAdversary(n_agents=2))
eval_env = ByzantineEnv(ToyGame(num_envs=40, episode_length=T), SingleAdversary(n_agents=2))

belief = CategoricalBelief(env.spec, env.prior)
# config = EnergyConfig(residual="soft", selection_coef=0.01, selection_decay_iters=N_ITERS // 2)
# config = EnergyConfig(residual="soft", tau=1.0, selection_coef=0.01, selection_decay_iters=N_ITERS // 2)
config = EnergyConfig(residual="soft", tau=0.3, selection_coef=0.1, selection_decay_iters=N_ITERS)
print(config)
algo = Energy(
    team=TeamPolicy(env.spec, belief_dim=belief.policy_dim, hidden_sizes=H),
    adversary=AdversaryPolicy(env.spec, hidden_sizes=H),
    belief=belief,
    critic=QCritic(env.spec, hidden_sizes=H),
    config=config,
)
# eval_score = analytical robustness (max 0.5). If the method works, it rises while energy falls.
algo.train(env, n_iters=N_ITERS, log_every=50,
           eval_fn=lambda a: analytical_robustness(eval_env, a.team, a.belief), eval_every=50)

clean = reward_stats(collect(eval_env, algo.team, algo.adversary, algo.belief))["reward_clean"]
result = evaluate_robustness(eval_env, algo.team, algo.belief, n_iters=1000,
                             config=GDAConfig(), hidden_sizes=H)     # same attacker as for GDA
robust = result["robust_reward"]
print(f"clean {clean:.3f} | robust {robust:.3f} | score {0.5 * clean + 0.5 * robust:.3f}")