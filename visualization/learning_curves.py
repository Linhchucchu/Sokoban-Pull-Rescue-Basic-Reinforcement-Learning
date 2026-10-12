"""
Plotting utilities for Reinforcement Learning training curves.
Generates Return Curves, Success Rates, and RMSE vs Oracle.
"""

from typing import List, Dict, Any, Optional, Tuple
import os
import matplotlib.pyplot as plt
import numpy as np

from visualization.ieee_style import apply_ieee_style, IEEE_DOUBLE_COL_WIDTH


class LearningCurveVisualizer:
    @staticmethod
    def rolling_window(data: List[float], window_size: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes rolling mean and standard deviation.
        """
        arr = np.array(data, dtype=np.float64)
        if len(arr) < window_size:
            return arr, np.zeros_like(arr)
        
        means = []
        stds = []
        for i in range(len(arr)):
            start = max(0, i - window_size + 1)
            sub = arr[start : i + 1]
            means.append(np.mean(sub))
            stds.append(np.std(sub))
        return np.array(means), np.array(stds)

    @classmethod
    def plot_comparison_curves(
        cls,
        results_dict: Dict[str, Dict[str, Any]],
        save_path: str,
        window_size: int = 100,
        title: str = "Comparative Learning Curves of Model-Free RL Algorithms",
    ):
        """
        results_dict: {algo_name: result_dict_from_train}
        Plots:
          1. Episodic Return (Rolling mean +/- std)
          2. Success Rate (%)
          3. Episode Length
          4. RMSE vs Oracle Q* (if available)
        """
        apply_ieee_style()
        fig, axes = plt.subplots(2, 2, figsize=(IEEE_DOUBLE_COL_WIDTH, 5.2))
        colors = {"Q-Learning": "#003366", "SARSA": "#B22222", "Expected SARSA": "#1B4D3E", "Monte Carlo": "#4B0082"}

        for name, res in results_dict.items():
            color = colors.get(name, None)
            episodes = np.arange(1, len(res["returns"]) + 1)

            # 1. Return
            r_mean, r_std = cls.rolling_window(res["returns"], window_size)
            axes[0, 0].plot(episodes, r_mean, label=name, color=color, linewidth=2)
            axes[0, 0].fill_between(episodes, r_mean - 0.5 * r_std, r_mean + 0.5 * r_std, color=color, alpha=0.15)

            # 2. Success Rate
            s_mean, _ = cls.rolling_window(res["success"], window_size)
            axes[0, 1].plot(episodes, s_mean * 100, label=name, color=color, linewidth=2)

            # 3. Episode Length
            l_mean, _ = cls.rolling_window(res["lengths"], window_size)
            axes[1, 0].plot(episodes, l_mean, label=name, color=color, linewidth=2)

            # 4. RMSE vs Oracle
            if "rmse" in res and res["rmse"]:
                rmse_eps, rmse_vals = zip(*res["rmse"])
                axes[1, 1].plot(rmse_eps, rmse_vals, label=name, color=color, linewidth=2, marker="o", markersize=3)

        # Formatting
        axes[0, 0].set_title("1. Mean Episodic Return", fontsize=10.5, weight="normal")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Return")
        axes[0, 0].grid(True, ls="--", alpha=0.5)
        axes[0, 0].legend()

        axes[0, 1].set_title(f"2. Task Completion Success Rate (Window = {window_size})", fontsize=10.5, weight="normal")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Success Rate (%)")
        axes[0, 1].set_ylim(-5, 105)
        axes[0, 1].grid(True, ls="--", alpha=0.5)
        axes[0, 1].legend()

        axes[1, 0].set_title("3. Average Steps to Goal (Episode Length)", fontsize=10.5, weight="normal")
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Steps")
        axes[1, 0].grid(True, ls="--", alpha=0.5)
        axes[1, 0].legend()

        axes[1, 1].set_title(r"4. RMSE vs Theoretical Oracle $Q^*$", fontsize=10.5, weight="normal")
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("RMSE")
        axes[1, 1].grid(True, ls="--", alpha=0.5)
        if axes[1, 1].get_legend_handles_labels()[0]:
            axes[1, 1].legend()

        plt.suptitle(title, fontsize=10.5, weight="normal", y=0.99)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
