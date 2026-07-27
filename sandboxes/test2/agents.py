from dataclasses import dataclass
import tensorflow as tf
from sandboxes.test2.networks.policy_coop import build_policy
from sandboxes.test2.networks.policy_adv import build_adversary
from sandboxes.test2.networks.value_net import build_value_net
from sandboxes.test2.networks.belief_net import build_belief

@dataclass
class Agents:
    policy: object
    adversary: object
    value: object
    belief: object
    opt_policy: object
    opt_adv: object
    opt_value: object
    opt_belief: object
    belief_cell: object   # GRUCell for single-step rollout
    belief_head: object   # sigmoid head

def build_agents(game, hidden=128, lr_policy=5e-4, lr_adv=1e-3, lr_critic=5e-4, lr_belief=5e-4):
    policy    = build_policy(game, hidden)
    adversary = build_adversary(game, hidden)
    value    = build_value_net(game, hidden)
    belief    = build_belief(game, hidden)
    return Agents(
        policy=policy, 
        adversary=adversary, 
        value=value, 
        belief=belief,
        opt_policy=tf.keras.optimizers.Adam(lr_policy),
        opt_adv=tf.keras.optimizers.Adam(lr_adv),
        opt_value=tf.keras.optimizers.Adam(lr_critic),
        opt_belief=tf.keras.optimizers.Adam(lr_belief),
        belief_cell=belief.get_layer(index=1).cell,
        belief_head=belief.get_layer("belief_out"),
    )