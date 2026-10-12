"""
Stage 3 Runner: Model-Free Tabular RL (Chapters 5, 6, 7).
Runs Monte Carlo, Stochastic Approximation, SARSA, and Q-Learning.
Compares SARSA vs Q-Learning, plots learning curves, and extracts snapshots at 100, 5000, and final iterations.
"""

import os
import sys
import shutil
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from sokoban_env.sokoban_pull_env import SokobanPullEnv
from algorithms.monte_carlo import MonteCarloControl
from algorithms.stochastic_approximation import StochasticApproximationAnalysis
from algorithms.temporal_difference import TemporalDifferenceControl
from visualization.david_silver_style import DavidSilverVisualizer
from visualization.learning_curves import LearningCurveVisualizer
from visualization.ieee_style import apply_ieee_style, IEEE_SINGLE_COL_WIDTH, IEEE_DOUBLE_COL_WIDTH


def run_stage3():
    print("=" * 80)
    print("STAGE 3: MODEL-FREE TABULAR RL (MONTE CARLO, ROBBINS-MONRO, TEMPORAL DIFFERENCE)")
    print("=" * 80)

    # -------------------------------------------------------------
    # 1. ROBBINS-MONRO STOCHASTIC APPROXIMATION
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print("PART 1: Stochastic Approximation (Robbins-Monro) & Learning Rate Schedules")
    print("-" * 60)
    sa_res = StochasticApproximationAnalysis.simulate_mean_estimation(
        true_mean=10.0, noise_std=5.0, num_steps=5000
    )
    for name, data in sa_res.items():
        print(f"   - {name:<26}: Estimate = {data['final_estimate']:.4f} (Error: {data['error']:.4f}), Sum(a): {data['sum_alpha']:.1f}, Sum(a^2): {data['sum_alpha_sq']:.1f}")

    # Plot Robbins-Monro convergence (IEEE Publication Style)
    apply_ieee_style()
    plt.figure(figsize=(4.5, 2.7))
    styles = {
        "Constant (alpha = 0.1)": ("#B22222", "-"),
        "Harmonic (alpha = 1/k)": ("#003366", "--"),
        "Polynomial (alpha = 1/k^0.6)": ("#1B4D3E", "-"),
    }
    for name, data in sa_res.items():
        color, ls = styles.get(name, (None, "-"))
        plt.plot(data["estimates"], label=name, color=color, linestyle=ls, linewidth=1.4)
    plt.axhline(10.0, color="#222222", linestyle=":", linewidth=1.2, label=r"True Expectation $\mathbb{E}[X] = 10.0$")
    plt.xlabel(r"Iteration Step $k$", fontsize=8.5)
    plt.ylabel(r"Estimate $w_k$", fontsize=8.5)
    plt.title(r"Robbins-Monro Convergence across $\alpha_k$ Schedules", fontsize=9.5, weight="bold")
    plt.grid(True, ls="--", alpha=0.4)
    plt.legend(framealpha=0.85, fontsize=7.5)
    plt.tight_layout()
    for folder in ["figures", "figure", os.path.join("output", "plots")]:
        sa_plot_path = os.path.join(PROJECT_ROOT, folder, "robbins_monro_convergence.png")
        os.makedirs(os.path.dirname(sa_plot_path), exist_ok=True)
        plt.savefig(sa_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   Saved plot to: figures/ & output/plots/robbins_monro_convergence.png")

    # -------------------------------------------------------------
    # 2. MONTE CARLO CONTROL (TABULAR)
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print("PART 2: Monte Carlo Learning on 5x5 Basic Map")
    print("-" * 60)
    map_5x5 = os.path.join(PROJECT_ROOT, "maps", "map_5x5_basic.txt")
    env_5x5 = SokobanPullEnv(map_5x5, mode="basic")
    
    # Load Oracle Q*
    oracle_q_path = os.path.join(PROJECT_ROOT, "data", "oracle_basic_5x5_Q.npy")
    if not os.path.exists(oracle_q_path):
        oracle_q_path = os.path.join(PROJECT_ROOT, "output", "tables", "oracle_basic_5x5_Q.npy")
    oracle_Q_5x5 = np.load(oracle_q_path) if os.path.exists(oracle_q_path) else None

    mc = MonteCarloControl(env_5x5, gamma=0.9, alpha=0.1, first_visit=True)
    mc_returns = []
    mc_success = []
    print("   Training Monte Carlo for 5,000 episodes...")
    for ep in range(1, 5001):
        r, solved = mc.train_episode_incremental()
        mc_returns.append(r)
        mc_success.append(1 if solved else 0)

    mc_success_rate = np.mean(mc_success[-500:]) * 100
    print(f"   Monte Carlo completed! Final 500-ep success rate: {mc_success_rate:.1f}%")

    # -------------------------------------------------------------
    # 3. TEMPORAL DIFFERENCE (Q-LEARNING VS SARSA)
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print("PART 3: Temporal Difference Learning (SARSA vs Q-Learning vs Expected SARSA)")
    print("-" * 60)
    num_episodes = 20000
    snapshot_eps = (10, 100, 500, 5000)

    # A. Q-Learning
    print(f"1. Training Q-Learning ({num_episodes} episodes)...")
    ql_agent = TemporalDifferenceControl(
        env_5x5,
        algo="q_learning",
        gamma=0.9,
        alpha=0.1,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.9997,
        oracle_Q=oracle_Q_5x5,
    )
    ql_res = ql_agent.train(num_episodes=num_episodes, log_interval=500, snapshot_episodes=snapshot_eps)
    print(f"   Time: {ql_res['elapsed_time']:.2f}s | Final success rate: {np.mean(ql_res['success'][-500:])*100:.1f}%")

    # B. SARSA
    print(f"2. Training SARSA ({num_episodes} episodes)...")
    sarsa_agent = TemporalDifferenceControl(
        env_5x5,
        algo="sarsa",
        gamma=0.9,
        alpha=0.1,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.9997,
        oracle_Q=oracle_Q_5x5,
    )
    sarsa_res = sarsa_agent.train(num_episodes=num_episodes, log_interval=500, snapshot_episodes=snapshot_eps)
    print(f"   Time: {sarsa_res['elapsed_time']:.2f}s | Final success rate: {np.mean(sarsa_res['success'][-500:])*100:.1f}%")

    # C. Expected SARSA
    print(f"3. Training Expected SARSA ({num_episodes} episodes)...")
    esarsa_agent = TemporalDifferenceControl(
        env_5x5,
        algo="expected_sarsa",
        gamma=0.9,
        alpha=0.1,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.9997,
        oracle_Q=oracle_Q_5x5,
    )
    esarsa_res = esarsa_agent.train(num_episodes=num_episodes, log_interval=500, snapshot_episodes=snapshot_eps)
    print(f"   Time: {esarsa_res['elapsed_time']:.2f}s | Final success rate: {np.mean(esarsa_res['success'][-500:])*100:.1f}%")

    # -------------------------------------------------------------
    # 4. MODEL-FREE TABULAR RL ON 7X7 HOOK MAP (|S| = 5,328, |A| = 5)
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print("PART 4: Model-Free Q-Learning on 7x7 Hook Map (|S| = 5,328, |A| = 5)")
    print("-" * 60)
    map_7x7 = os.path.join(PROJECT_ROOT, "maps", "map_7x7_hook.txt")
    env_7x7 = SokobanPullEnv(map_7x7, mode="special")
    
    num_episodes_7x7 = 10000
    print(f"Training Q-Learning on 7x7 Hook Map ({num_episodes_7x7} episodes)...")
    ql_7x7 = TemporalDifferenceControl(
        env_7x7,
        algo="q_learning",
        gamma=0.9,
        alpha=0.1,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.9995,
    )
    ql_7x7_res = ql_7x7.train(num_episodes=num_episodes_7x7, log_interval=2000)
    succ_7x7 = np.mean(ql_7x7_res["success"][-500:]) * 100
    steps_7x7 = np.mean(ql_7x7_res["lengths"][-500:])
    ret_7x7 = np.mean(ql_7x7_res["returns"][-500:])
    print(f"   7x7 Q-Learning Success: {succ_7x7:.1f}% | Steps: {steps_7x7:.1f} | Return: {ret_7x7:.1f}")

    # -------------------------------------------------------------
    # 5. VISUALIZE PROGRESSION AT SNAPSHOTS (10, 100, 500, 5000)
    # -------------------------------------------------------------
    print("\nVisualizing Q-Learning progression at 10, 100, 500, and 5,000 episodes (David Silver Style)...")
    viz = DavidSilverVisualizer(env_5x5)
    
    # Extract history_V and history_pi at snapshots
    hist_V = [ql_res["snapshots"][ep]["V"] for ep in snapshot_eps]
    hist_pi = [ql_res["snapshots"][ep]["policy"] for ep in snapshot_eps]

    for folder in ["figures", "figure", os.path.join("output", "plots")]:
        prog_plot_path = os.path.join(PROJECT_ROOT, folder, "q_learning_progression_100_5000_opt.png")
        os.makedirs(os.path.dirname(prog_plot_path), exist_ok=True)
        viz.plot_david_silver_iterations(
            history_V=hist_V,
            history_pi=hist_pi,
            box_pos=env_5x5.initial_box_pos,
            iterations_to_plot=list(snapshot_eps),
            save_path=prog_plot_path,
            title="Q-Learning Value Function and Policy Evolution across Episode Milestones",
        )
        prog_new_path = os.path.join(PROJECT_ROOT, folder, "q_learning_progression_10_100_500_5000.png")
        shutil.copyfile(prog_plot_path, prog_new_path)
    print(f"   Saved plot to: figures/ & output/plots/q_learning_progression_10_100_500_5000.png")

    # Plot comparison curves
    comp_results = {
        "Q-Learning": ql_res,
        "SARSA": sarsa_res,
        "Expected SARSA": esarsa_res,
    }
    for folder in ["figures", "figure", os.path.join("output", "plots")]:
        curve_plot_path = os.path.join(PROJECT_ROOT, folder, "td_learning_curves_comparison.png")
        os.makedirs(os.path.dirname(curve_plot_path), exist_ok=True)
        LearningCurveVisualizer.plot_comparison_curves(
            results_dict=comp_results,
            save_path=curve_plot_path,
            window_size=200,
            title="Head-to-Head Comparison: Q-Learning vs SARSA vs Expected SARSA",
        )
    print(f"   Saved plot to: figures/ & output/plots/td_learning_curves_comparison.png")

    print("\n" + "=" * 80)
    print("COMPLETED STAGE 3: MODEL-FREE TABULAR RL!")
    print("=" * 80)


if __name__ == "__main__":
    run_stage3()
