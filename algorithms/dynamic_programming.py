"""
Chapter 4: Dynamic Programming Algorithms.
Implements:
1. Value Iteration (VI)
2. Policy Iteration (PI)
3. Truncated Policy Iteration (TPI)
Captures snapshots for David Silver style visualizations and exports Oracle tables.
"""

from typing import Tuple, List, Dict, Any, Optional
import os
import time
import numpy as np


class DynamicProgramming:
    def __init__(self, P: np.ndarray, R: np.ndarray, gamma: float = 0.9):
        """
        P: (num_states, num_actions, num_states)
        R: (num_states, num_actions)
        """
        self.P = P
        self.R = R
        self.gamma = gamma
        self.num_states, self.num_actions, _ = P.shape

    def compute_q_table(self, V: np.ndarray) -> np.ndarray:
        """
        Q(s, a) = R(s, a) + gamma * sum_{s'} P(s'|s,a) * V(s')
        """
        expected_future = np.einsum("san,n->sa", self.P, V)
        return self.R + self.gamma * expected_future

    def extract_greedy_policy(self, Q: np.ndarray) -> np.ndarray:
        """
        pi(s) = argmax_a Q(s, a)
        """
        return np.argmax(Q, axis=1)

    def value_iteration(
        self,
        theta: float = 1e-7,
        max_iters: int = 1000,
    ) -> Dict[str, Any]:
        """
        Value Iteration:
        V_{k+1}(s) = max_a [ R(s, a) + gamma * sum_{s'} P(s'|s,a) * V_k(s') ]
        """
        start_time = time.perf_counter()
        V = np.zeros(self.num_states, dtype=np.float64)
        history_V = [V.copy()]
        history_pi = [self.extract_greedy_policy(self.compute_q_table(V))]
        errors = []

        for k in range(max_iters):
            Q = self.compute_q_table(V)
            V_next = np.max(Q, axis=1)
            err = np.max(np.abs(V_next - V))
            errors.append(err)
            V = V_next

            # Record history
            history_V.append(V.copy())
            history_pi.append(self.extract_greedy_policy(Q))

            if err < theta:
                break

        elapsed = time.perf_counter() - start_time
        final_Q = self.compute_q_table(V)
        final_pi = self.extract_greedy_policy(final_Q)

        return {
            "V": V,
            "Q": final_Q,
            "policy": final_pi,
            "num_iterations": len(errors),
            "total_backups": len(errors) * self.num_states * self.num_actions,
            "elapsed_time": elapsed,
            "errors": errors,
            "history_V": history_V,
            "history_pi": history_pi,
        }

    def policy_iteration(
        self,
        theta: float = 1e-7,
        max_iters: int = 500,
    ) -> Dict[str, Any]:
        """
        Policy Iteration:
        Alternates between complete Policy Evaluation and Greedy Policy Improvement.
        """
        start_time = time.perf_counter()
        policy = np.zeros(self.num_states, dtype=int)
        V = np.zeros(self.num_states, dtype=np.float64)
        history_V = [V.copy()]
        history_pi = [policy.copy()]
        total_eval_steps = 0

        for it in range(max_iters):
            # 1. Policy Evaluation
            # Form P_pi and R_pi
            P_pi = self.P[np.arange(self.num_states), policy]
            R_pi = self.R[np.arange(self.num_states), policy]

            # Solve (I - gamma * P_pi) V = R_pi
            I = np.eye(self.num_states)
            V = np.linalg.solve(I - self.gamma * P_pi, R_pi)
            total_eval_steps += 1
            history_V.append(V.copy())

            # 2. Policy Improvement
            Q = self.compute_q_table(V)
            new_policy = np.argmax(Q, axis=1)
            history_pi.append(new_policy.copy())

            # Check stability
            if np.array_equal(new_policy, policy):
                policy = new_policy
                break
            policy = new_policy

        elapsed = time.perf_counter() - start_time
        final_Q = self.compute_q_table(V)

        return {
            "V": V,
            "Q": final_Q,
            "policy": policy,
            "num_iterations": it + 1,
            "total_eval_steps": total_eval_steps,
            "elapsed_time": elapsed,
            "history_V": history_V,
            "history_pi": history_pi,
        }

    def truncated_policy_iteration(
        self,
        m: int = 5,
        theta: float = 1e-7,
        max_iters: int = 500,
    ) -> Dict[str, Any]:
        """
        Truncated Policy Iteration:
        Runs exactly m steps of iterative policy evaluation, then improves policy.
        """
        start_time = time.perf_counter()
        V = np.zeros(self.num_states, dtype=np.float64)
        policy = np.zeros(self.num_states, dtype=int)
        errors = []
        total_backups = 0

        for it in range(max_iters):
            # m steps of Policy Evaluation
            P_pi = self.P[np.arange(self.num_states), policy]
            R_pi = self.R[np.arange(self.num_states), policy]

            for _ in range(m):
                V = R_pi + self.gamma * np.dot(P_pi, V)
                total_backups += self.num_states

            # Policy Improvement
            Q = self.compute_q_table(V)
            new_policy = np.argmax(Q, axis=1)
            total_backups += self.num_states * self.num_actions

            err = np.max(np.abs(np.max(Q, axis=1) - V))
            errors.append(err)

            if np.array_equal(new_policy, policy) and err < theta:
                policy = new_policy
                break
            policy = new_policy

        elapsed = time.perf_counter() - start_time
        final_Q = self.compute_q_table(V)

        return {
            "m": m,
            "V": V,
            "Q": final_Q,
            "policy": policy,
            "num_iterations": it + 1,
            "total_backups": total_backups,
            "elapsed_time": elapsed,
            "errors": errors,
        }

    @staticmethod
    def save_tables(
        V: np.ndarray,
        Q: np.ndarray,
        policy: np.ndarray,
        all_states: List[Tuple],
        output_dir: str,
        prefix: str = "oracle_basic",
    ):
        """
        Saves V, Q, Policy in both CSV and NPY formats.
        """
        os.makedirs(output_dir, exist_ok=True)

        # 1. Save NPY
        np.save(os.path.join(output_dir, f"{prefix}_V.npy"), V)
        np.save(os.path.join(output_dir, f"{prefix}_Q.npy"), Q)
        np.save(os.path.join(output_dir, f"{prefix}_policy.npy"), policy)

        # 2. Save Human-readable CSV for V & Policy
        csv_v_path = os.path.join(output_dir, f"{prefix}_V_policy.csv")
        with open(csv_v_path, "w", encoding="utf-8") as f:
            f.write("state_idx,state_tuple,value_V,optimal_action,optimal_action_name\n")
            action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "PULL"}
            for idx, st in enumerate(all_states):
                act = policy[idx]
                act_name = action_names.get(act, str(act))
                f.write(f"{idx},\"{st}\",{V[idx]:.4f},{act},{act_name}\n")

        # 3. Save CSV for Q-Table
        csv_q_path = os.path.join(output_dir, f"{prefix}_Q_table.csv")
        with open(csv_q_path, "w", encoding="utf-8") as f:
            headers = ["state_idx", "state_tuple"] + [f"Q_a{a}" for a in range(Q.shape[1])]
            f.write(",".join(headers) + "\n")
            for idx, st in enumerate(all_states):
                q_vals = [f"{Q[idx, a]:.4f}" for a in range(Q.shape[1])]
                f.write(f"{idx},\"{st}\"," + ",".join(q_vals) + "\n")
