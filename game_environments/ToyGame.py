import numpy as np

class ToyGame:
    """B copies of the toy game, stepped in parallel."""
    def __init__(self, B, ep_len=10, p_no_adv=0.5):
        self.N = 2
        self.n_actions = 2
        self.n_states = 2
        self.gamma = 0.99
        self.ep_len = ep_len
        self.p_no_adv = p_no_adv
        self.B = B
        self.reset()

    def reset(self):
        self.t = 0
        self.state = np.random.randint(self.n_states, size=self.B)  # (B,)
        return self._obs()

    def _obs(self):
        # one-hot state per env, shared across agents: (B, N, n_states)
        oh = np.zeros((self.B, self.n_states), np.float32)
        oh[np.arange(self.B), self.state] = 1.0
        return np.repeat(oh[:, None, :], self.N, axis=1)

    def step(self, actions):
        a = actions.astype(int)
        agree = (a[:, 0] == a[:, 1])
        in_s1 = (self.state == 0)
        r = np.where(in_s1, agree.astype(np.float32), (~agree).astype(np.float32))

        # transition (per your spec):
        #   s1 & agree  -> s2   |   s1 & differ -> s1
        #   s2 & differ -> s2   |   s2 & agree  -> s1
        next_state = np.where(
            in_s1,
            np.where(agree, 1, 0),      # from s1: agree->s2(1), differ->stay s1(0)
            np.where(agree, 0, 1),      # from s2: agree->s1(0), differ->stay s2(1)
        )
        self.state = next_state

        self.t += 1
        done = self.t >= self.ep_len
        return self._obs(), r, done

