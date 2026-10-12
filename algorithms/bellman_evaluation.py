"""
Chapter 2: Bellman Equation & Policy Evaluation.
Computes V_pi via:
1. Exact Matrix Inversion: V_pi = (I - gamma * P_pi)^(-1) * R_pi
2. Iterative Policy Evaluation: V_{k+1} = T^pi V_k
Also computes Q_pi and greedy policy pi'.
"""

from typing import Tuple, List, Dict
import numpy as np


class BellmanPolicyEvaluation:
    def __init__(self, P: np.ndarray, R: np.ndarray, gamma: float = 0.9):
        """
        P: (num_states, num_actions, num_states)
        R: (num_states, num_actions)
        gamma: discount factor
        """
        self.P = P
        self.R = R
        self.gamma = gamma
        self.num_states, self.num_actions, _ = P.shape

    def get_mrp_matrices(self, policy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collapses MDP into a Markov Reward Process (MRP) / Markov Chain under policy pi.
        policy: (num_states, num_actions) probability distribution
        Returns:
            P_pi: (num_states, num_states)
            R_pi: (num_states,)
        """
        # P_pi[s, s'] = sum_a pi(a|s) * P[s, a, s']
        P_pi = np.einsum("sa,san->sn", policy, self.P)
        # R_pi[s] = sum_a pi(a|s) * R[s, a]
        R_pi = np.einsum("sa,sa->s", policy, self.R)
        return P_pi, R_pi

    def solve_matrix_inversion(self, policy: np.ndarray) -> np.ndarray:
        """
        Exact closed-form solution: V_pi = (I - gamma * P_pi)^(-1) * R_pi
        """
        P_pi, R_pi = self.get_mrp_matrices(policy)
        I = np.eye(self.num_states)
        A = I - self.gamma * P_pi
        V_pi = np.linalg.solve(A, R_pi)
        return V_pi

    def solve_iterative(
        self,
        policy: np.ndarray,
        theta: float = 1e-7,
        max_iters: int = 1000,
    ) -> Tuple[np.ndarray, List[float], List[np.ndarray]]:
        """
        Iterative Policy Evaluation:
        V_{k+1} = R_pi + gamma * P_pi * V_k
        Returns:
            V_pi: final value table
            errors: list of infinity norm errors ||V_{k+1} - V_k||_inf
            history_V: snapshots of V across iterations
        """
        P_pi, R_pi = self.get_mrp_matrices(policy)
        V = np.zeros(self.num_states, dtype=np.float64)
        errors = []
        history_V = [V.copy()]

        for k in range(max_iters):
            V_next = R_pi + self.gamma * np.dot(P_pi, V)
            err = np.max(np.abs(V_next - V))
            errors.append(err)
            V = V_next
            history_V.append(V.copy())

            if err < theta:
                break

        return V, errors, history_V

    def compute_q_table(self, V: np.ndarray) -> np.ndarray:
        """
        Q(s, a) = R(s, a) + gamma * sum_{s'} P(s'|s,a) * V(s')
        """
        # (num_states, num_actions)
        expected_future = np.einsum("san,n->sa", self.P, V)
        Q = self.R + self.gamma * expected_future
        return Q

    def get_greedy_policy(self, Q: np.ndarray) -> np.ndarray:
        """
        Extracts greedy deterministic policy pi(s) = argmax_a Q(s, a)
        """
        return np.argmax(Q, axis=1)
