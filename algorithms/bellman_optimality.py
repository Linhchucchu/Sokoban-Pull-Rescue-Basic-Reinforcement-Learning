"""
Chapter 3: Bellman Optimality Equation (BOE).
Analyzes the Bellman Optimality Operator T* and its Banach contraction mapping property:
||T* U - T* V||_inf <= gamma * ||U - V||_inf
"""

from typing import Tuple, Dict
import numpy as np


class BellmanOptimalityAnalysis:
    def __init__(self, P: np.ndarray, R: np.ndarray):
        """
        P: (num_states, num_actions, num_states)
        R: (num_states, num_actions)
        """
        self.P = P
        self.R = R
        self.num_states, self.num_actions, _ = P.shape

    def bellman_optimality_operator(self, V: np.ndarray, gamma: float) -> np.ndarray:
        """
        T* V(s) = max_a [ R(s, a) + gamma * sum_{s'} P(s'|s,a) * V(s') ]
        """
        expected_future = np.einsum("san,n->sa", self.P, V)
        Q = self.R + gamma * expected_future
        return np.max(Q, axis=1)

    def verify_contraction_property(
        self,
        gamma: float,
        num_trials: int = 50,
    ) -> Dict[str, float]:
        """
        Generates random value vectors U, V and measures the contraction ratio:
        ratio = ||T* U - T* V||_inf / ||U - V||_inf
        Should always be <= gamma!
        """
        max_ratio = 0.0
        ratios = []

        for _ in range(num_trials):
            U = np.random.uniform(-50, 100, size=self.num_states)
            V = np.random.uniform(-50, 100, size=self.num_states)
            diff_input = np.max(np.abs(U - V))
            if diff_input < 1e-6:
                continue

            TU = self.bellman_optimality_operator(U, gamma)
            TV = self.bellman_optimality_operator(V, gamma)
            diff_output = np.max(np.abs(TU - TV))

            ratio = diff_output / diff_input
            ratios.append(ratio)
            if ratio > max_ratio:
                max_ratio = ratio

        return {
            "gamma": gamma,
            "max_ratio": max_ratio,
            "mean_ratio": float(np.mean(ratios)),
            "is_contraction": bool(max_ratio <= gamma + 1e-9),
        }
