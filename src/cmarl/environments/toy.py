"""Two-agent toy coordination game (Han et al., 2022), as used in EIR-MAPPO.

Two states, two actions per agent, both agents fully observe the state.

    State 0 (XNOR): reward 1 if the agents AGREE,  else 0.
    State 1 (XOR) : reward 1 if the agents DIFFER, else 0.

Transitions (agreeing flips the state, differing keeps it):
    s0: agree -> s1,  differ -> s0
    s1: agree -> s0,  differ -> s1

Observation (same for both agents, same layout as the official code minus
the agent ID, which TeamPolicy adds itself):
    [state, last_state, last_action_agent0, last_action_agent1]
At reset: last_state = state, last actions = 0 (as in the official code).

Robust value under attack: 0.5 per step. Clean optimum: 1.0.
"""

import torch

from cmarl.environments.base import EnvSpec, StepResult, VecEnv


class ToyGame(VecEnv):

    def __init__(self, num_envs: int, episode_length: int = 100, device: str = "cpu"):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.spec = EnvSpec(
            n_agents=2,
            n_actions=2,
            obs_dim=4,        # state, last state, 2 last actions
            state_dim=4,      # the critic sees the same vector
            episode_length=episode_length,
        )
        self.t = 0
        self.state = torch.zeros(num_envs, dtype=torch.long, device=self.device)        # (B,)
        self.last_state = torch.zeros(num_envs, dtype=torch.long, device=self.device)   # (B,)
        self.last_actions = torch.zeros(num_envs, 2, dtype=torch.long, device=self.device)  # (B, 2)

    def reset(self) -> StepResult:
        self.t = 0
        self.state = torch.randint(0, 2, (self.num_envs,), device=self.device)
        self.last_state = self.state.clone()
        self.last_actions = torch.zeros(self.num_envs, 2, dtype=torch.long, device=self.device)
        zeros = torch.zeros(self.num_envs, device=self.device)
        done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return StepResult(obs=self._obs(), state=self._state(), reward=zeros, done=done)

    def step(self, actions: torch.Tensor) -> StepResult:
        # actions: (B, 2) long, the executed actions
        agree = actions[:, 0] == actions[:, 1]
        in_s0 = self.state == 0
        reward = torch.where(in_s0, agree, ~agree).float()

        self.last_state = self.state
        self.last_actions = actions.clone()
        self.state = torch.where(agree, 1 - self.state, self.state)

        self.t += 1
        done = torch.full((self.num_envs,), self.t >= self.spec.episode_length,
                          dtype=torch.bool, device=self.device)
        return StepResult(obs=self._obs(), state=self._state(), reward=reward, done=done)

    # ----- helpers -----

    def _state(self) -> torch.Tensor:
        """[state, last_state, a0, a1] as floats: (B, 4)."""
        return torch.cat([
            self.state.unsqueeze(1),
            self.last_state.unsqueeze(1),
            self.last_actions,
        ], dim=1).float()

    def _obs(self) -> torch.Tensor:
        """Both agents see the same vector: (B, N, 4)."""
        return self._state().unsqueeze(1).expand(-1, self.spec.n_agents, -1)

    def worst_case_actions(self, team_probs: torch.Tensor) -> torch.Tensor:
        """White-box adversary used by the official code to select checkpoints.
        Agent i, as adversary, knows its partner's action probabilities this step:
            s0 (team wants to agree) : play the opposite of the partner's most likely action
            s1 (team wants to differ): copy the partner's most likely action
        team_probs (B, 2, A) -> actions (B, 2); entry i = what agent i plays as adversary.
        """
        partner_mode = team_probs.argmax(-1).flip(1)          # (B, 2) partner's most likely action
        in_s0 = (self.state == 0).unsqueeze(1)                 # (B, 1)
        return torch.where(in_s0, 1 - partner_mode, partner_mode)