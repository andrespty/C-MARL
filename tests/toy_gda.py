import torch
from cmarl.environments.toy import ToyGame
from cmarl.byzantine.priors import AdaptiveSingleAdversary, SingleAdversary
from cmarl.byzantine.wrapper import ByzantineEnv
from cmarl.agents.belief import CategoricalBelief
from cmarl.agents.policy import TeamPolicy, AdversaryPolicy
from cmarl.agents.critic import VCritic
from cmarl.algorithms.base import reward_stats
from cmarl.algorithms.gda import GDA, GDAConfig
from cmarl.evaluate import analytical_robustness, evaluate_robustness
from cmarl.rollout import collect

T = 100
H = (128, 128, 128)
INTERVAL = 10
N_ITERS = 10_000
SEEDS = [0, 1, 2, 3, 4]


def run(seed: int) -> dict:
    torch.manual_seed(seed)
    env = ByzantineEnv(
        ToyGame(num_envs=40, episode_length=T),
        AdaptiveSingleAdversary(n_agents=2, temperature=1 / T, interval=INTERVAL))
    eval_env = ByzantineEnv(ToyGame(num_envs=40, episode_length=T), SingleAdversary(n_agents=2))

    belief = CategoricalBelief(env.spec, env.prior)
    config = GDAConfig(team_interval=INTERVAL)          # defaults follow the official code
    algo = GDA(
        team=TeamPolicy(env.spec, belief_dim=belief.policy_dim, hidden_sizes=H),
        adversary=AdversaryPolicy(env.spec, hidden_sizes=H),
        belief=belief,
        critic=VCritic(env.spec, belief_dim=belief.policy_dim, hidden_sizes=H),
        config=config,
    )

    def checkpoint_score(a) -> float:
        """Paper: keep the model with improved robustness without decreasing cooperation."""
        clean = reward_stats(collect(eval_env, a.team, a.adversary, a.belief))["reward_clean"]
        return 0.5 * clean + 0.5 * analytical_robustness(eval_env, a.team, a.belief)

    algo.train(env, n_iters=N_ITERS, log_every=100,
               teacher_forcing=True, eval_fn=checkpoint_score, eval_every=25)
    algo.load_best()                                    # checkpoint selection

    clean = reward_stats(collect(eval_env, algo.team, algo.adversary, algo.belief))["reward_clean"]
    result = evaluate_robustness(eval_env, algo.team, algo.belief, n_iters=1000,
                                 config=config, hidden_sizes=H)
    robust = result["robust_reward"]
    return {"seed": seed, "clean": clean, "robust": robust,
            "best_ckpt_score": algo.best_score, "score": 0.5 * clean + 0.5 * robust}


if __name__ == "__main__":
    results = [run(s) for s in SEEDS]
    for r in results:
        print(" | ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items()))
    scores = torch.tensor([r["score"] for r in results])
    print(f"score: {scores.mean():.3f} ± {scores.std():.3f}")