"""
Discount Factor Ablation Study (gamma = 0.8 vs 0.9 vs 0.99 vs 1.0).
Analyzes contraction rate, convergence speed, horizon, and infinite loop risks when gamma = 1.0.
"""

import os
import sys
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
from algorithms.dynamic_programming import DynamicProgramming
from algorithms.temporal_difference import TemporalDifferenceControl
from visualization.learning_curves import LearningCurveVisualizer
from visualization.ieee_style import apply_ieee_style, IEEE_SINGLE_COL_WIDTH


def run_discount_ablation():
    apply_ieee_style()
    print("=" * 80)
    print("DISCOUNT FACTOR ABLATION STUDY: gamma = 0.8 vs 0.9 vs 0.99 vs 1.0")
    print("=" * 80)

    map_path = os.path.join(PROJECT_ROOT, "maps", "map_5x5_basic.txt")
    env = SokobanPullEnv(map_path, mode="basic")
    P, R = env.get_transition_model()

    gammas = [0.8, 0.9, 0.99, 1.0]
    dp_results = {}
    td_results = {}

    print("\n1. Dynamic Programming (Value Iteration) Analysis across Discount Factors:")
    for g in gammas:
        dp = DynamicProgramming(P, R, gamma=g)
        vi_res = dp.value_iteration(theta=1e-7, max_iters=2000)
        dp_results[g] = vi_res
        v_range = (vi_res["V"].min(), vi_res["V"].max())
        print(f"   gamma = {g:4.2f} -> Iterations to converge: {vi_res['num_iterations']:4d} | V* range: [{v_range[0]:.2f}, {v_range[1]:.2f}] | Time: {vi_res['elapsed_time']*1000:.2f} ms")

    # Plot DP Convergence comparison (IEEE Single Column Width)
    palette = ["#003366", "#1B4D3E", "#D95F02", "#7570B3"]
    plt.figure(figsize=(IEEE_SINGLE_COL_WIDTH, 2.7))
    for idx, g in enumerate(gammas):
        plt.plot(dp_results[g]["errors"], label=rf"$\gamma = {g}$ ({dp_results[g]['num_iterations']} iters)", color=palette[idx], linewidth=1.5)
    plt.yscale("log")
    plt.xlabel("Value Iteration Step", fontsize=8.5)
    plt.ylabel(r"Bellman Error $\|V_{k+1} - V_k\|_\infty$", fontsize=8.5)
    plt.title(r"Value Iteration Convergence vs Discount Factor $\gamma$", fontsize=9.2, weight="bold")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=7.5)
    for folder in ["figures", os.path.join("output", "plots")]:
        dp_gamma_plot = os.path.join(PROJECT_ROOT, folder, "dp_gamma_convergence.png")
        os.makedirs(os.path.dirname(dp_gamma_plot), exist_ok=True)
        plt.savefig(dp_gamma_plot, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   Saved plot to: figures/ & output/plots/dp_gamma_convergence.png")

    # 2. TD Learning comparison across gammas (10,000 episodes)
    print("\n2. Training Q-Learning (10,000 episodes) for each gamma value:")
    for g in gammas:
        td = TemporalDifferenceControl(env, algo="q_learning", gamma=g, alpha=0.1, epsilon_decay=0.9995)
        res = td.train(num_episodes=10000)
        td_results[f"gamma = {g}"] = res
        succ_final = np.mean(res["success"][-500:]) * 100
        print(f"   gamma = {g:4.2f} -> Final 500-ep Success: {succ_final:.1f}% | Time: {res['elapsed_time']:.2f}s")

    for folder in ["figures", os.path.join("output", "plots")]:
        td_gamma_plot = os.path.join(PROJECT_ROOT, folder, "td_gamma_ablation_comparison.png")
        os.makedirs(os.path.dirname(td_gamma_plot), exist_ok=True)
        LearningCurveVisualizer.plot_comparison_curves(
            td_results,
            save_path=td_gamma_plot,
            window_size=200,
            title="Q-Learning Performance Comparison across Discount Factor $\\gamma$",
        )
    print(f"   Saved plot to: figures/ & output/plots/td_gamma_ablation_comparison.png")

    print("\n" + "=" * 80)
    print("COMPLETED DISCOUNT FACTOR ABLATION STUDY!")
    print("=" * 80)


if __name__ == "__main__":
    run_discount_ablation()
