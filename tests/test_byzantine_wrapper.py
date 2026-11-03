import torch
from cmarl.environments.toy import ToyGame
from cmarl.byzantine.priors import SingleAdversary
from cmarl.byzantine.wrapper import ByzantineEnv

env = ByzantineEnv(ToyGame(num_envs=4), SingleAdversary(n_agents=2))
r = env.reset()
team = torch.zeros(4, 2, dtype=torch.long)
adv  = torch.ones(4, 2, dtype=torch.long)
r = env.step(team, adv)
print(r.theta)
print(r.executed)                         # equals theta here: 1 where compromised
assert torch.equal(r.executed, r.theta)