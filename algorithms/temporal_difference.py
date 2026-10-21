"""
Chapter 7: Temporal Difference Learning (Tabular).
Implements:
1. Q-Learning (Off-policy TD control)
2. SARSA (On-policy TD control)
3. Expected SARSA
Optimized with integer state transitions for ultra-fast training (1M episodes in 2-3 mins).
"""

from typing import Tuple, List, Dict, Any, Optional
import time
import numpy as np


class TemporalDifferenceControl:
    def __init__(
        self,
        env,
        algo: str = "q_learning",  # "q_learning", "sarsa", "expected_sarsa"
        gamma: float = 0.9,
        alpha: float = 0.1,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.9995,
        oracle_Q: Optional[np.ndarray] = None,
    ):
        self.env = env
        self.algo = algo
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.oracle_Q = oracle_Q

        self.num_states = env.num_states
        self.num_actions = env.num_actions

        self.Q = np.zeros((self.num_states, self.num_actions), dtype=np.float64)
        self.N = np.zeros((self.num_states, self.num_actions), dtype=np.int32)

        # Precompute deterministic lookup table: next_idx[s, a] and reward[s, a]
        # This makes 1M episodes blazingly fast in pure NumPy!
        self._build_fast_transition_cache()

    def _build_fast_transition_cache(self):
        self.cache_next_s = np.zeros((self.num_states, self.num_actions), dtype=np.int32)
        self.cache_r = np.zeros((self.num_states, self.num_actions), dtype=np.float32)
        self.cache_term = np.zeros((self.num_states, self.num_actions), dtype=bool)

        for s_idx, state in enumerate(self.env.all_states):
            for a in range(self.num_actions):
                next_st, r, term, _, _ = self.env.simulate_step(state, a)
                self.cache_next_s[s_idx, a] = self.env.state_to_idx(next_st)
                self.cache_r[s_idx, a] = r
                self.cache_term[s_idx, a] = term

        if self.env.mode == "basic":
            init_st = (
                self.env.initial_agent_pos[0],
                self.env.initial_agent_pos[1],
                self.env.initial_box_pos[0],
                self.env.initial_box_pos[1],
            )
        else:
            init_st = (
                self.env.initial_agent_pos[0],
                self.env.initial_agent_pos[1],
                self.env.initial_box_pos[0],
                self.env.initial_box_pos[1],
                1,
                0,
            )
        self.initial_s_idx = self.env.state_to_idx(init_st)

    def select_action(self, s: int) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        q_s = self.Q[s]
        # Fast tie-breaking: if all equal (unvisited), pick random action; else argmax
        if np.all(q_s == q_s[0]):
            return np.random.randint(self.num_actions)
        return int(np.argmax(q_s))

    def train(
        self,
        num_episodes: int = 10000,
        max_steps_per_episode: int = 100,
        log_interval: int = 500,
        snapshot_episodes: Tuple[int, ...] = (100, 5000),
    ) -> Dict[str, Any]:
        """
        Runs TD training loop.
        """
        start_time = time.perf_counter()
        returns_history = []
        lengths_history = []
        success_history = []
        rmse_history = []
        snapshots = {}

        recent_returns = []
        recent_success = []

        for ep in range(1, num_episodes + 1):
            s = self.initial_s_idx
            ep_return = 0.0
            steps = 0
            solved = False

            if self.algo == "sarsa":
                a = self.select_action(s)

            for step in range(max_steps_per_episode):
                steps += 1
                if self.algo != "sarsa":
                    a = self.select_action(s)

                self.N[s, a] += 1
                next_s = self.cache_next_s[s, a]
                r = self.cache_r[s, a]
                done = self.cache_term[s, a]

                ep_return += r

                # TD Update
                if done:
                    target = r
                else:
                    if self.algo == "q_learning":
                        target = r + self.gamma * np.max(self.Q[next_s])
                    elif self.algo == "sarsa":
                        next_a = self.select_action(next_s)
                        target = r + self.gamma * self.Q[next_s, next_a]
                    elif self.algo == "expected_sarsa":
                        # Expected value under epsilon-greedy policy
                        q_next = self.Q[next_s]
                        best_a = np.argmax(q_next)
                        probs = np.full(self.num_actions, self.epsilon / self.num_actions)
                        probs[best_a] += (1.0 - self.epsilon)
                        target = r + self.gamma * np.sum(probs * q_next)

                td_error = target - self.Q[s, a]
                self.Q[s, a] += self.alpha * td_error

                s = next_s
                if self.algo == "sarsa" and not done:
                    a = next_a

                if done:
                    if r > 50.0:  # Goal reward
                        solved = True
                    break

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            returns_history.append(ep_return)
            lengths_history.append(steps)
            success_history.append(1 if solved else 0)
            recent_returns.append(ep_return)
            recent_success.append(1 if solved else 0)

            # Compute RMSE vs Oracle Q* if available
            if self.oracle_Q is not None and (ep % log_interval == 0 or ep in snapshot_episodes):
                rmse = np.sqrt(np.mean((self.Q - self.oracle_Q) ** 2))
                rmse_history.append((ep, rmse))

            # Record snapshots
            if ep in snapshot_episodes or ep == num_episodes:
                snapshots[ep] = {
                    "Q": self.Q.copy(),
                    "V": np.max(self.Q, axis=1),
                    "policy": np.argmax(self.Q, axis=1),
                    "epsilon": self.epsilon,
                }

        elapsed = time.perf_counter() - start_time
        return {
            "algo": self.algo,
            "num_episodes": num_episodes,
            "elapsed_time": elapsed,
            "Q": self.Q,
            "V": np.max(self.Q, axis=1),
            "policy": np.argmax(self.Q, axis=1),
            "returns": returns_history,
            "lengths": lengths_history,
            "success": success_history,
            "rmse": rmse_history,
            "snapshots": snapshots,
        }
