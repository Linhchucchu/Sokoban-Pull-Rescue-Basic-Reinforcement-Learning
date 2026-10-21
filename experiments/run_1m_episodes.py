"""
1 Million Episodes Large-Scale Scaling Experiment on Map 7x7 Hook (Setting 4).
Runs Tabular Q-Learning for 1,000,000 episodes using ultra-fast NumPy integer cache.
Tracks Q-table saturation, state-action coverage N(s, a), and RMSE vs Oracle Q*.
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
from visualization.ieee_style import apply_ieee_style, IEEE_DOUBLE_COL_WIDTH


def run_1m(total_episodes=None):
    if total_episodes is None:
        import argparse
        parser = argparse.ArgumentParser(description="Large-Scale Tabular Q-Learning on Map 7x7 Hook")
        parser.add_argument("--episodes", type=int, default=50000, help="Number of episodes to train (Default: 50,000 - Sweet Spot)")
        args, _ = parser.parse_known_args()
        total_episodes = args.episodes

    print("=" * 80)
    print(f"LARGE-SCALE SCALING EXPERIMENT: {total_episodes:,} EPISODES ON MAP 7x7 HOOK (SETTING 4)")
    print("=" * 80)

    map_path = os.path.join(PROJECT_ROOT, "maps", "map_7x7_hook.txt")
    env = SokobanPullEnv(map_path, mode="special")
    print(f"Map: 7x7 Hook | State Space: |S| = {env.num_states} | Action Space: |A| = {env.num_actions}")

    # Load Oracle Q*
    oracle_q_path = os.path.join(PROJECT_ROOT, "data", "oracle_special_7x7_hook_Q.npy")
    if not os.path.exists(oracle_q_path):
        oracle_q_path = os.path.join(PROJECT_ROOT, "output", "tables", "oracle_special_7x7_hook_Q.npy")
    oracle_Q = np.load(oracle_q_path) if os.path.exists(oracle_q_path) else None
    if oracle_Q is not None:
        print(f"   [+] Loaded theoretical Oracle Q* ({oracle_Q.shape}) for RMSE calculation!")

    chunk_size = max(5000, total_episodes // 10)
    num_chunks = max(1, total_episodes // chunk_size)
    eps_decay = 0.99995 if total_episodes <= 100000 else 0.999995

    agent = TemporalDifferenceControl(
        env,
        algo="q_learning",
        gamma=0.9,
        alpha=0.08,
        epsilon_start=1.0,
        epsilon_min=0.02,
        epsilon_decay=eps_decay,
        oracle_Q=oracle_Q,
    )

    print(f"\nStarting training for {total_episodes:,} episodes ({num_chunks} chunks, {chunk_size:,} eps each)...")
    start_total = time.perf_counter()

    chunk_success_rates = []
    chunk_returns = []
    chunk_rmses = []
    episodes_axis = []

    for chunk in range(1, num_chunks + 1):
        t0 = time.perf_counter()
        res = agent.train(num_episodes=chunk_size, log_interval=chunk_size, snapshot_episodes=())
        t1 = time.perf_counter()

        succ = np.mean(res["success"][-1000:]) * 100
        ret = np.mean(res["returns"][-1000:])
        curr_ep = chunk * chunk_size
        episodes_axis.append(curr_ep)
        chunk_success_rates.append(succ)
        chunk_returns.append(ret)

        rmse = 0.0
        if oracle_Q is not None:
            rmse = float(np.sqrt(np.mean((agent.Q - oracle_Q) ** 2)))
            chunk_rmses.append(rmse)

        visited_pairs = np.count_nonzero(agent.N)
        total_pairs = env.num_states * env.num_actions
        coverage = (visited_pairs / total_pairs) * 100

        print(f"   [Chunk {chunk:2d}/{num_chunks}] Ep {curr_ep:7,d} | Return: {ret:6.1f} | Success: {succ:5.1f}% | RMSE: {rmse:5.2f} | Coverage (s,a): {coverage:5.1f}% | Time: {t1-t0:4.1f}s")

    total_elapsed = time.perf_counter() - start_total
    print(f"\n=> Completed {total_episodes:,} episodes in {total_elapsed:.1f} seconds ({total_elapsed/60:.2f} minutes)!")

    # Save final Q-table
    for target_dir in [os.path.join(PROJECT_ROOT, "data"), os.path.join(PROJECT_ROOT, "output", "tables")]:
        os.makedirs(target_dir, exist_ok=True)
        np.save(os.path.join(target_dir, "q_learning_1m_special_Q.npy"), agent.Q)
    print(f"   [+] Saved Q-table to: data/ & output/tables/q_learning_1m_special_Q.npy")

    # Plot 1M Scaling Curve (IEEE Double Column Width)
    apply_ieee_style()
    fig, axes = plt.subplots(1, 3, figsize=(IEEE_DOUBLE_COL_WIDTH, 2.5))
    axes[0].plot(episodes_axis, chunk_returns, color="#003366", marker="o", linewidth=1.5, markersize=3.5)
    axes[0].set_title("1. Episodic Return", fontsize=9.0, weight="bold")
    axes[0].set_xlabel(r"Episodes $N$", fontsize=8.5)
    axes[0].set_ylabel("Episodic Return", fontsize=8.5)
    axes[0].grid(True, ls="--", alpha=0.4)

    axes[1].plot(episodes_axis, chunk_success_rates, color="#1B4D3E", marker="s", linewidth=1.5, markersize=3.5)
    axes[1].set_title(r"2. Success Rate (\%)", fontsize=9.0, weight="bold")
    axes[1].set_xlabel(r"Episodes $N$", fontsize=8.5)
    axes[1].set_ylabel(r"Success Rate (\%)", fontsize=8.5)
    axes[1].set_ylim(-5, 105)
    axes[1].grid(True, ls="--", alpha=0.4)

    if chunk_rmses:
        axes[2].plot(episodes_axis, chunk_rmses, color="#B22222", marker="^", linewidth=1.5, markersize=3.5)
        axes[2].set_title(r"3. RMSE vs Oracle $Q^*$", fontsize=9.0, weight="bold")
        axes[2].set_xlabel(r"Episodes $N$", fontsize=8.5)
        axes[2].set_ylabel("RMSE", fontsize=8.5)
        axes[2].grid(True, ls="--", alpha=0.4)

    plt.suptitle(rf"Tabular Q-Learning Scaling Experiment ({total_episodes:,} Episodes) on Setting 4", fontsize=9.8, weight="bold", y=0.98)
    plt.tight_layout()
    for folder in ["figures", os.path.join("output", "plots")]:
        plot_path = os.path.join(PROJECT_ROOT, folder, "q_learning_1m_episodes_scaling.png")
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   Saved plot to: figures/ & output/plots/q_learning_1m_episodes_scaling.png")

    print("\n" + "=" * 80)
    print("COMPLETED LARGE-SCALE EPISODES SCALING EXPERIMENT!")
    print("=" * 80)


if __name__ == "__main__":
    run_1m()
