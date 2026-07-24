from dataclasses import dataclass
import tensorflow as tf
from networks import build_policy, build_adversary, build_critic, build_belief

@dataclass
class Agents:
    policy: object
    adversary: object
    critic: object
    belief: object
    opt_policy: object
    opt_adv: object
    opt_critic: object
    opt_belief: object
    belief_cell: object   # GRUCell for single-step rollout
    belief_head: object   # sigmoid head

def build_agents(cfg, hidden=128, lr_policy=5e-4, lr_adv=1e-3, lr_critic=5e-4, lr_belief=5e-4):
    policy    = build_policy(cfg, hidden)
    adversary = build_adversary(cfg, hidden)
    critic    = build_critic(cfg, hidden)
    belief    = build_belief(cfg, hidden)
    return Agents(
        policy=policy, 
        adversary=adversary, 
        critic=critic, 
        belief=belief,
        opt_policy=tf.keras.optimizers.Adam(lr_policy),
        opt_adv=tf.keras.optimizers.Adam(lr_adv),
        opt_critic=tf.keras.optimizers.Adam(lr_critic),
        opt_belief=tf.keras.optimizers.Adam(lr_belief),
        belief_cell=belief.get_layer(index=1).cell,
        belief_head=belief.get_layer("belief_out"),
    )