from dataclasses import dataclass
import tensorflow as tf
from networks.policy import build_policy
from networks.adversary import build_adversary
from networks.value_net import build_value_net


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
    belief_cell: object   # recurrent cell for single-step rollout (GRUCell)
    belief_head: object   # belief output layer (sigmoid or softmax head)


def build_agents(game, belief_module, hidden=128, lr_policy=5e-4, lr_adv=1e-3,
                 lr_critic=5e-4, lr_belief=5e-4):
    """belief_module is one of beliefs.factored_sigmoid / beliefs.categorical_eirmappo.
    It determines both the belief network's architecture and the policy's
    belief-input width (belief_module.POLICY_INPUT_DIM)."""
    belief     = belief_module.build_belief(game, hidden)
    belief_dim = belief_module.POLICY_INPUT_DIM(game, hidden)
    policy     = build_policy(game, hidden, belief_dim)
    adversary  = build_adversary(game, hidden)
    value      = build_value_net(game, hidden)
    return Agents(
        policy=policy,
        adversary=adversary,
        value=value,
        belief=belief,
        opt_policy=tf.keras.optimizers.legacy.Adam(lr_policy),
        opt_adv=tf.keras.optimizers.legacy.Adam(lr_adv),
        opt_value=tf.keras.optimizers.legacy.Adam(lr_critic),
        opt_belief=tf.keras.optimizers.legacy.Adam(lr_belief),
        belief_cell=belief_module.get_recurrent_cell(belief),
        belief_head=belief_module.get_output_head(belief),
    )
