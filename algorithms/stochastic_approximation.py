"""
Chapter 6: Stochastic Approximation (Robbins-Monro Theorem).
Theoretical and empirical foundation for Temporal Difference updates.
Tests convergence under Robbins-Monro conditions:
sum alpha_k = inf  and  sum alpha_k^2 < inf
"""

from typing import Dict, List, Tuple, Any
import numpy as np


class StochasticApproximationAnalysis:
    @staticmethod
    def simulate_mean_estimation(
        true_mean: float = 10.0,
        noise_std: float = 5.0,
        num_steps: int = 10000,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Simulates estimating E[X] via stochastic approximation:
        w_{k+1} = w_k + alpha_k (X_k - w_k)
        under 3 learning rate schedules:
        1. Constant alpha = 0.1
        2. Harmonic decay alpha_k = 1/k (Robbins-Monro)
        3. Polynomial decay alpha_k = 1 / k^0.6
        """
        np.random.seed(42)
        noise = np.random.normal(0, noise_std, size=num_steps)
        samples = true_mean + noise

        schedules = {
            "Constant (alpha = 0.1)": lambda k: 0.1,
            "Harmonic (alpha = 1/k)": lambda k: 1.0 / (k + 1),
            "Polynomial (alpha = 1/k^0.6)": lambda k: 1.0 / ((k + 1) ** 0.6),
        }

        results = {}
        for name, alpha_fn in schedules.items():
            estimates = []
            alphas = []
            w = 0.0  # Initial estimate

            for k in range(num_steps):
                a_k = alpha_fn(k)
                alphas.append(a_k)
                x_k = samples[k]
                w = w + a_k * (x_k - w)
                estimates.append(w)

            sum_a = sum(alphas)
            sum_a2 = sum(a**2 for a in alphas)

            results[name] = {
                "estimates": estimates,
                "final_estimate": w,
                "error": abs(w - true_mean),
                "sum_alpha": sum_a,
                "sum_alpha_sq": sum_a2,
                "satisfies_rm": (sum_a > 100 and sum_a2 < 100),
            }

        return results
