from dataclasses import dataclass

@dataclass
class Config:
    N: int           # number of agents
    n_actions: int   
    n_states: int      
    ep_len: int      # timesteps per episode (paper uses 100)
    gamma: float
    p_no_adv: float