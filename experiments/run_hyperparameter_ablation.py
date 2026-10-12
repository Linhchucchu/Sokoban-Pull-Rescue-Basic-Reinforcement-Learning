"""
Hyperparameter Ablation Study for Tabular Reinforcement Learning on Sokoban.
Conducts quantitative experiments on:
1. Training Episodes Scaling: N in [1,000, 5,000, 10,000, 20,000, 50,000] (Trade-off: Time vs Accuracy/RMSE)
2. Learning Rate alpha in [0.02, 0.05, 0.10, 0.20] (Convergence speed vs Stability)
3. Exploration Decay epsilon_decay in [0.999, 0.9995, 0.9998] (Exploration vs Exploitation)
"""

import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from sokoban_env.sokoban_pull_env import SokobanPullEnv
from algorithms.temporal_difference import TemporalDifferenceControl
from visualization.learning_curves import LearningCurveVisualizer
from visualization.ieee_style import apply_ieee_style, IEEE_DOUBLE_COL_WIDTH


def run_ablation():
    print("=" * 80)
    print("HYPERPARAMETER ABLATION STUDY (EPISODES N, LEARNING RATE ALPHA, EPSILON DECAY)")
    print("=" * 80)

    map_path = os.path.join(PROJECT_ROOT, "maps", "map_5x5_basic.txt")
    env = SokobanPullEnv(map_path, mode="basic")
    print(f"Test Environment: 5x5 Basic Map | |S| = {env.num_states}, |A| = {env.num_actions}")

    # Load Oracle Q*
    oracle_q_path = os.path.join(PROJECT_ROOT, "data", "oracle_basic_5x5_Q.npy")
    if not os.path.exists(oracle_q_path):
        oracle_q_path = os.path.join(PROJECT_ROOT, "output", "tables", "oracle_basic_5x5_Q.npy")
    oracle_Q = np.load(oracle_q_path) if os.path.exists(oracle_q_path) else None
    if oracle_Q is not None:
        print(f"[+] Loaded theoretical Oracle Q* ({oracle_Q.shape}) for exact RMSE computation!")

    # -------------------------------------------------------------------------
    # ABLATION 1: EPISODE BUDGET SCALING (N in [1k, 5k, 10k, 20k, 50k])
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("1. EPISODE BUDGET SCALING: N in [1k, 5k, 10k, 20k, 50k]")
    print("-" * 70)

    episode_budgets = [1000, 5000, 10000, 20000, 50000]
    ep_results = []

    for N in episode_budgets:
        eps_decay = 0.995 if N <= 5000 else (0.9995 if N <= 20000 else 0.9998)
        agent = TemporalDifferenceControl(
            env,
            algo="q_learning",
            gamma=0.9,
            alpha=0.1,
            epsilon_start=1.0,
            epsilon_min=0.05,
            epsilon_decay=eps_decay,
            oracle_Q=oracle_Q,
        )
        t0 = time.perf_counter()
        res = agent.train(num_episodes=N, log_interval=N+1)
        elapsed = time.perf_counter() - t0

        final_succ = np.mean(res["success"][-min(500, N):]) * 100
        final_ret = np.mean(res["returns"][-min(500, N):])
        rmse = float(np.sqrt(np.mean((agent.Q - oracle_Q) ** 2))) if oracle_Q is not None else 0.0
        visited = np.count_nonzero(agent.N)
        coverage = (visited / (env.num_states * env.num_actions)) * 100

        ep_results.append({
            "N": N,
            "time_sec": elapsed,
            "success_rate": final_succ,
            "mean_return": final_ret,
            "rmse": rmse,
            "coverage": coverage
        })
        print(f"   N = {N:6,d} eps -> Time: {elapsed:5.2f}s | Success: {final_succ:5.1f}% | Return: {final_ret:6.1f} | RMSE: {rmse:5.2f} | Coverage: {coverage:4.1f}%")

    # Plot Episode Ablation (IEEE Publication Standard)
    apply_ieee_style()
    fig, axes = plt.subplots(1, 3, figsize=(IEEE_DOUBLE_COL_WIDTH, 2.3))
    n_vals = [r["N"] for r in ep_results]

    axes[0].plot(n_vals, [r["success_rate"] for r in ep_results], color="#1B4D3E", marker="o", linewidth=1.5, markersize=4)
    axes[0].set_title("Success Rate (%)", fontsize=9.0, weight="bold")
    axes[0].set_xlabel(r"Episode Budget $N$", fontsize=8.5)
    axes[0].set_ylabel("Success Rate (%)", fontsize=8.5)
    axes[0].set_ylim(-5, 105)
    axes[0].grid(True, ls="--", alpha=0.4)

    axes[1].plot(n_vals, [r["rmse"] for r in ep_results], color="#B22222", marker="s", linewidth=1.5, markersize=4)
    axes[1].set_title(r"RMSE vs Oracle $Q^*$", fontsize=9.0, weight="bold")
    axes[1].set_xlabel(r"Episode Budget $N$", fontsize=8.5)
    axes[1].set_ylabel("RMSE", fontsize=8.5)
    axes[1].grid(True, ls="--", alpha=0.4)

    axes[2].plot(n_vals, [r["time_sec"] for r in ep_results], color="#003366", marker="^", linewidth=1.5, markersize=4)
    axes[2].set_title("Wall-Clock Time (s)", fontsize=9.0, weight="bold")
    axes[2].set_xlabel(r"Episode Budget $N$", fontsize=8.5)
    axes[2].set_ylabel("Time (s)", fontsize=8.5)
    axes[2].grid(True, ls="--", alpha=0.4)

    plt.suptitle(r"Episode Budget $N$ Scaling: Training Time vs Solution Quality", fontsize=9.8, weight="bold", y=0.98)
    plt.tight_layout()
    for folder in ["figures", "figure", os.path.join("output", "plots")]:
        ep_plot = os.path.join(PROJECT_ROOT, folder, "ablation_episodes_scaling.png")
        os.makedirs(os.path.dirname(ep_plot), exist_ok=True)
        plt.savefig(ep_plot, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   [+] Saved plot to: figures/ & output/plots/ablation_episodes_scaling.png")

    # -------------------------------------------------------------------------
    # ABLATION 2: LEARNING RATE alpha in [0.02, 0.05, 0.10, 0.20]
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("2. LEARNING RATE ABLATION: alpha in [0.02, 0.05, 0.10, 0.20]")
    print("-" * 70)

    alphas = [0.02, 0.05, 0.10, 0.20]
    alpha_curves = {}

    for a in alphas:
        agent = TemporalDifferenceControl(
            env,
            algo="q_learning",
            gamma=0.9,
            alpha=a,
            epsilon_start=1.0,
            epsilon_min=0.05,
            epsilon_decay=0.9995,
            oracle_Q=oracle_Q,
        )
        res = agent.train(num_episodes=15000, log_interval=15001)
        alpha_curves[f"alpha = {a}"] = res
        succ = np.mean(res["success"][-500:]) * 100
        rmse = float(np.sqrt(np.mean((agent.Q - oracle_Q) ** 2))) if oracle_Q is not None else 0.0
        print(f"   alpha = {a:4.2f} -> Final 500-ep Success: {succ:5.1f}% | RMSE: {rmse:5.2f} | Time: {res['elapsed_time']:.2f}s")

    for folder in ["figures", "figure", os.path.join("output", "plots")]:
        alpha_plot = os.path.join(PROJECT_ROOT, folder, "ablation_learning_rate_comparison.png")
        os.makedirs(os.path.dirname(alpha_plot), exist_ok=True)
        LearningCurveVisualizer.plot_comparison_curves(
            alpha_curves,
            save_path=alpha_plot,
            window_size=200,
            title="Q-Learning Convergence Speed across Learning Rates $\\alpha$",
        )
    print(f"   [+] Saved plot to: figures/ & output/plots/ablation_learning_rate_comparison.png")

    # -------------------------------------------------------------------------
    # ABLATION 3: EXPLORATION DECAY epsilon_decay in [0.999, 0.9995, 0.9998]
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("3. EXPLORATION DECAY ABLATION (EXPLORATION VS EXPLOITATION)")
    print("-" * 70)

    decays = [0.999, 0.9995, 0.9998]
    decay_curves = {}

    for d in decays:
        agent = TemporalDifferenceControl(
            env,
            algo="q_learning",
            gamma=0.9,
            alpha=0.1,
            epsilon_start=1.0,
            epsilon_min=0.05,
            epsilon_decay=d,
            oracle_Q=oracle_Q,
        )
        res = agent.train(num_episodes=15000, log_interval=15001)
        decay_curves[f"eps_decay = {d}"] = res
        succ = np.mean(res["success"][-500:]) * 100
        print(f"   eps_decay = {d:6.4f} -> Final 500-ep Success: {succ:5.1f}% | Time: {res['elapsed_time']:.2f}s")

    for folder in ["figures", "figure", os.path.join("output", "plots")]:
        decay_plot = os.path.join(PROJECT_ROOT, folder, "ablation_epsilon_decay_comparison.png")
        os.makedirs(os.path.dirname(decay_plot), exist_ok=True)
        LearningCurveVisualizer.plot_comparison_curves(
            decay_curves,
            save_path=decay_plot,
            window_size=200,
            title="Q-Learning Exploration Dynamics across Exploration Decay $\\varepsilon_{\\text{decay}}$",
        )
    print(f"   [+] Saved plot to: figures/ & output/plots/ablation_epsilon_decay_comparison.png")

    print("\n" + "=" * 80)
    print("COMPLETED HYPERPARAMETER ABLATION STUDY!")
    print("=" * 80)


if __name__ == "__main__":
    run_ablation()
