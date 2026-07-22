import numpy as np

class ToyGame:
    """B copies of the toy game, stepped in parallel."""
    def __init__(self, cfg, B):
        self.cfg = cfg
        self.B = B
        self.reset()

    def reset(self):
        self.t = 0
        self.state = np.random.randint(self.cfg.n_states, size=self.B)  # (B,)
        return self._obs()

    def _obs(self):
        # one-hot state per env, shared across agents: (B, N, n_states)
        oh = np.zeros((self.B, self.cfg.n_states), np.float32)
        oh[np.arange(self.B), self.state] = 1.0
        return np.repeat(oh[:, None, :], self.cfg.N, axis=1)

    def step(self, actions):
        # actions: (B, N) perturbed actions
        a = actions.astype(int)
        agree = (a[:, 0] == a[:, 1])                    # (B,) N=2 specialization
        in_s1 = (self.state == 0)
        r = np.where(in_s1, agree.astype(np.float32),   # s1: reward if agree
                            (~agree).astype(np.float32)) # s2: reward if differ
        self.state = np.where(agree, 0, 1)              # transition
        self.t += 1
        done = self.t >= self.cfg.ep_len
        return self._obs(), r, done

