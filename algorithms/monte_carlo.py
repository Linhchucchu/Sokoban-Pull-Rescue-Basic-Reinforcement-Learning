"""
Chapter 5: Monte Carlo Learning (Tabular).
Supports:
1. Non-incremental MC (batch average) vs Incremental MC (online update)
2. First-visit vs Every-visit
3. Epsilon-greedy exploration
"""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np


class MonteCarloControl:
    def __init__(
        self,
        env,
        gamma: float = 0.9,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.9995,
        alpha: Optional[float] = None,  # If None, uses 1/N(s, a) (Sample Average); if float, uses constant step-size
        first_visit: bool = True,
    ):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.first_visit = first_visit

        self.num_states = env.num_states
        self.num_actions = env.num_actions

        self.Q = np.zeros((self.num_states, self.num_actions), dtype=np.float64)
        self.N = np.zeros((self.num_states, self.num_actions), dtype=np.int32)
        # Returns accumulator for non-incremental
        self.returns_sum = np.zeros((self.num_states, self.num_actions), dtype=np.float64)

    def select_action(self, state_idx: int) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        q_vals = self.Q[state_idx]
        max_q = np.max(q_vals)
        best_actions = np.where(q_vals == max_q)[0]
        return np.random.choice(best_actions)

    def generate_episode(self, max_steps: int = 100) -> List[Tuple[int, int, float]]:
        """
        Rolls out an episode following epsilon-greedy policy.
        Returns list of (state_idx, action, reward).
        """
        obs, info = self.env.reset()
        st_idx = info["state_idx"]
        episode = []

        for _ in range(max_steps):
            act = self.select_action(st_idx)
            next_obs, reward, term, trunc, next_info = self.env.step(act)
            next_st_idx = next_info["state_idx"]
            episode.append((st_idx, act, reward))
            st_idx = next_st_idx
            if term or trunc:
                break

        return episode

    def train_episode_incremental(self, max_steps: int = 100) -> Tuple[float, bool]:
        """
        Incremental Monte Carlo update.
        """
        episode = self.generate_episode(max_steps)
        total_return = sum(r for _, _, r in episode)
        is_solved = self.env.is_solved

        G = 0.0
        visited = set()
        # Traverse in reverse
        for t in reversed(range(len(episode))):
            s_t, a_t, r_t = episode[t]
            G = r_t + self.gamma * G

            if self.first_visit and (s_t, a_t) in visited:
                continue
            visited.add((s_t, a_t))

            self.N[s_t, a_t] += 1
            if self.alpha is None:
                step_size = 1.0 / self.N[s_t, a_t]
            else:
                step_size = self.alpha

            self.Q[s_t, a_t] += step_size * (G - self.Q[s_t, a_t])

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return total_return, is_solved
