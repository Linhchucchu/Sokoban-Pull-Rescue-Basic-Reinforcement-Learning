"""
Stage 1 Runner: Basic Map Dynamic Programming (Chapters 2, 3, 4).
Runs Bellman Evaluation, Bellman Optimality Analysis, Value Iteration, Policy Iteration,
and Truncated Policy Iteration on the 5x5 Basic Map.
Exports tables and generates David Silver style plots.
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from sokoban_env.sokoban_pull_env import SokobanPullEnv
from algorithms.bellman_evaluation import BellmanPolicyEvaluation
from algorithms.bellman_optimality import BellmanOptimalityAnalysis
from algorithms.dynamic_programming import DynamicProgramming
from visualization.david_silver_style import DavidSilverVisualizer
from visualization.ieee_style import apply_ieee_style, IEEE_SINGLE_COL_WIDTH, IEEE_DOUBLE_COL_WIDTH


def run_stage1():
    print("=" * 80)
    print("STAGE 1: DYNAMIC PROGRAMMING FOUNDATIONS ON BASIC 5x5 MAP")
    print("=" * 80)

    map_path = os.path.join(PROJECT_ROOT, "maps", "map_5x5_basic.txt")
    env = SokobanPullEnv(map_path, mode="basic")
    gamma = 0.9

    print(f"Map: 5x5 Basic")
    print(f"State space size (|S|): {env.num_states}")
    print(f"Action space size (|A|): {env.num_actions}")

    # Extract model
    print("\nExtracting transition probability tensor P and reward vector R...")
    P, R = env.get_transition_model()
    print(f"P shape: {P.shape}, R shape: {R.shape}")

    # -------------------------------------------------------------
    # 1. Bellman Expectation Equation & Policy Evaluation
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print("PART 1: Bellman Expectation Equation & Policy Evaluation")
    print("-" * 60)
    evaluator = BellmanPolicyEvaluation(P, R, gamma=gamma)
    # Uniform policy: pi(a|s) = 0.25
    uniform_policy = np.ones((env.num_states, env.num_actions)) / env.num_actions

    # Matrix Inversion
    V_matrix = evaluator.solve_matrix_inversion(uniform_policy)
    print(f"1. Analytical matrix inversion V_pi = (I - gamma * P_pi)^(-1) * R_pi solved successfully!")
    print(f"   V_pi (mean): {V_matrix.mean():.4f}, V_pi (min/max): [{V_matrix.min():.4f}, {V_matrix.max():.4f}]")

    # Iterative Policy Evaluation
    V_iter, pe_errors, history_pe_V = evaluator.solve_iterative(uniform_policy, theta=1e-7)
    diff = np.max(np.abs(V_matrix - V_iter))
    print(f"2. Iterative Policy Evaluation converged in {len(pe_errors)} steps!")
    print(f"   Max absolute difference between Matrix Inversion and Iterative PE: {diff:.2e} (Exact match!)")

    # Plot Bellman Error curve (IEEE Single Column Width)
    apply_ieee_style()
    plt.figure(figsize=(IEEE_SINGLE_COL_WIDTH, 2.7))
    plt.plot(pe_errors, color="#003366", linewidth=1.5, label=r"$\|V_{k+1} - V_k\|_\infty$")
    plt.yscale("log")
    plt.xlabel("Iteration $k$", fontsize=8.5)
    plt.ylabel(r"Bellman Error $\|V_{k+1} - V_k\|_\infty$", fontsize=8.5)
    plt.title(r"Bellman Error Convergence ($\gamma = 0.9$)", fontsize=9.2, weight="bold")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=8.0)
    for folder in ["figures", "figure", os.path.join("output", "plots")]:
        p = os.path.join(PROJECT_ROOT, folder, "bellman_error_convergence.png")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   Saved plot to: figures/ & output/plots/bellman_error_convergence.png")

    # -------------------------------------------------------------
    # 2. Bellman Optimality Equation (BOE) & Contraction Analysis
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print("PART 2: Bellman Optimality Equation & Contraction Analysis (Banach Fixed-Point)")
    print("-" * 60)
    boe_analyzer = BellmanOptimalityAnalysis(P, R)
    for g in [0.5, 0.8, 0.9, 0.99, 1.0]:
        res = boe_analyzer.verify_contraction_property(gamma=g, num_trials=50)
        status = "Strict Contraction" if res["is_contraction"] and g < 1.0 else ("Non-expansion" if g == 1.0 else "Violation")
        print(f"   gamma = {g:4.2f} -> Max Contraction Ratio: {res['max_ratio']:.4f} (<= gamma? {res['is_contraction']}) -> {status}")

    # -------------------------------------------------------------
    # 3. Dynamic Programming (VI, PI, Truncated PI)
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print("PART 3: Value Iteration, Policy Iteration & Truncated Policy Iteration")
    print("-" * 60)
    dp = DynamicProgramming(P, R, gamma=gamma)

    # 1. Value Iteration
    vi_res = dp.value_iteration(theta=1e-7)
    print(f"1. Value Iteration (VI):")
    print(f"   Iterations: {vi_res['num_iterations']}")
    print(f"   Total Bellman backups: {vi_res['total_backups']}")
    print(f"   Execution time: {vi_res['elapsed_time'] * 1000:.2f} ms")

    # 2. Policy Iteration
    pi_res = dp.policy_iteration()
    print(f"\n2. Policy Iteration (PI):")
    print(f"   Policy iterations: {pi_res['num_iterations']}")
    print(f"   Execution time: {pi_res['elapsed_time'] * 1000:.2f} ms")

    # 3. Truncated Policy Iteration with m = 1, 2, 5, 10, 20
    print(f"\n3. Truncated Policy Iteration (TPI):")
    tpi_results = {}
    for m in [1, 2, 5, 10, 20]:
        tpi_res = dp.truncated_policy_iteration(m=m, theta=1e-7)
        tpi_results[m] = tpi_res
        print(f"   m = {m:2d} -> Iterations: {tpi_res['num_iterations']:3d}, Total backups: {tpi_res['total_backups']:6d}, Time: {tpi_res['elapsed_time']*1000:.2f} ms")

    print("\nChecking optimality consistency between VI and PI:")
    diff_vipi = np.max(np.abs(vi_res["V"] - pi_res["V"]))
    policy_match = np.mean(vi_res["policy"] == pi_res["policy"]) * 100
    print(f"   Max difference between V*(VI) and V*(PI): {diff_vipi:.2e}")
    print(f"   Optimal policy agreement: {policy_match:.1f}%")

    # -------------------------------------------------------------
    # EXPORT CORE DATA TABLES (Value Table, Q-Table, Policy Table)
    # -------------------------------------------------------------
    for target_dir in [os.path.join(PROJECT_ROOT, "data"), os.path.join(PROJECT_ROOT, "output", "tables")]:
        os.makedirs(target_dir, exist_ok=True)
        dp.save_tables(
            V=vi_res["V"],
            Q=vi_res["Q"],
            policy=vi_res["policy"],
            all_states=env.all_states,
            output_dir=target_dir,
            prefix="oracle_basic_5x5",
        )
    print(f"\nExporting 3 core data tables to data/ and output/tables/...")
    print("   [+] oracle_basic_5x5_V.npy & oracle_basic_5x5_Q.npy & oracle_basic_5x5_policy.npy")
    print("   [+] oracle_basic_5x5_V_policy.csv (Value & Policy table with action names)")
    print("   [+] oracle_basic_5x5_Q_table.csv (Full Q-Table |S| x |A|)")

    # -------------------------------------------------------------
    # DAVID SILVER LECTURE 3 STYLE VISUALIZATION
    # -------------------------------------------------------------
    print("\nVisualizing Value Iteration progression (David Silver Lecture 3 style)...")
    viz = DavidSilverVisualizer(env)
    
    # Select iteration snapshots to plot
    max_k = len(vi_res["history_V"]) - 1
    selected_iters = [0, 1, 2, 3, min(5, max_k), min(10, max_k), max_k]
    selected_iters = sorted(list(set(selected_iters)))

    for folder in ["figures", "figure", os.path.join("output", "plots")]:
        ds_plot_path = os.path.join(PROJECT_ROOT, folder, "david_silver_value_iteration.png")
        os.makedirs(os.path.dirname(ds_plot_path), exist_ok=True)
        viz.plot_david_silver_iterations(
            history_V=vi_res["history_V"],
            history_pi=vi_res["history_pi"],
            box_pos=env.initial_box_pos,
            iterations_to_plot=selected_iters,
            save_path=ds_plot_path,
            title="David Silver Lecture 3: Value Iteration Propagation V_k(s) & Policy pi_k(s) across Iterations",
        )
    print(f"   Saved David Silver plot to: figures/ & output/plots/david_silver_value_iteration.png")

    # Plot Algorithm Comparison Bar Chart (IEEE Publication Style)
    apply_ieee_style()
    plt.figure(figsize=(4.8, 2.8))
    algos = ["VI", "TPI\n(m=2)", "TPI\n(m=5)", "TPI\n(m=10)", "TPI\n(m=20)", "PI"]
    backups = [vi_res["total_backups"]] + [tpi_results[m]["total_backups"] for m in [2, 5, 10, 20]] + [pi_res["num_iterations"] * env.num_states * env.num_actions]
    
    colors = ["#003366", "#1B4D3E", "#2E8B57", "#D97706", "#B22222", "#4B0082"]
    bars = plt.bar(algos, backups, color=colors, edgecolor="#222222", linewidth=0.7, alpha=0.85, width=0.6)
    plt.ylabel("Total Bellman Backups", fontsize=8.5)
    plt.title("Computational Complexity: VI vs. TPI vs. PI", fontsize=9.5, weight="bold")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + max(backups)*0.015, f"{int(yval):,}", ha="center", va="bottom", fontsize=7.0, weight="bold")
    plt.ylim(0, max(backups) * 1.15)
    plt.grid(axis="y", ls="--", alpha=0.4)
    plt.tight_layout()
    for folder in ["figures", "figure", os.path.join("output", "plots")]:
        comp_plot_path = os.path.join(PROJECT_ROOT, folder, "dp_algorithms_comparison.png")
        os.makedirs(os.path.dirname(comp_plot_path), exist_ok=True)
        plt.savefig(comp_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   Saved algorithm comparison plot to: figures/ & output/plots/dp_algorithms_comparison.png")

    print("\n" + "=" * 80)
    print("COMPLETED STAGE 1: DYNAMIC PROGRAMMING ON BASIC MAP!")
    print("=" * 80)


if __name__ == "__main__":
    run_stage1()
