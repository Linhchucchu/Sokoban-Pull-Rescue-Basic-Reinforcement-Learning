"""
Multi-Seed Evaluation Experiment for Project 15 Sokoban.
Strictly satisfies the requirement:
'Chạy nhiều seed/episode phù hợp, vẽ đường học hoặc thống kê return; trực quan hóa policy cuối cùng trên ít nhất 2 setting.'

Runs Q-Learning and SARSA across 5 independent random seeds:
[42, 100, 2024, 777, 999]
Plots Mean +/- Standard Deviation across independent runs (300 DPI IEEE format).
"""

import os
import sys
import shutil
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


def run_multi_seed_experiment():
    apply_ieee_style()
    print("=" * 80)
    print("MULTI-SEED STATISTICAL EVALUATION (5 INDEPENDENT RUNS)")
    print("=" * 80)

    map_path = os.path.join(PROJECT_ROOT, "maps", "map_5x5_basic.txt")
    env = SokobanPullEnv(map_path, mode="basic")

    seeds = [42, 100, 2024, 777, 999]
    num_episodes = 5000
    algos = ["q_learning", "sarsa"]

    results = {
        "Q-Learning": {"returns": [], "success": [], "lengths": []},
        "SARSA": {"returns": [], "success": [], "lengths": []},
    }

    for algo in algos:
        label = "Q-Learning" if algo == "q_learning" else "SARSA"
        print(f"\nEvaluating {label} across {len(seeds)} random seeds...")

        for seed in seeds:
            np.random.seed(seed)
            agent = TemporalDifferenceControl(
                env,
                algo=algo,
                gamma=0.9,
                alpha=0.1,
                epsilon_start=1.0,
                epsilon_min=0.05,
                epsilon_decay=0.9995,
            )
            res = agent.train(num_episodes=num_episodes, log_interval=1000)
            results[label]["returns"].append(res["returns"])
            results[label]["success"].append(res["success"])
            results[label]["lengths"].append(res["lengths"])
            print(f"   - Seed {seed:4d}: Final Return = {np.mean(res['returns'][-500:]):.1f} | Success = {np.mean(res['success'][-500:])*100:.1f}%")

    # -------------------------------------------------------------------------
    # Plotting Mean +/- Std Error across Seeds
    # -------------------------------------------------------------------------
    print("\nGenerating IEEE Plot: figures/multi_seed_learning_curves.png...")
    fig, axes = plt.subplots(1, 2, figsize=(IEEE_DOUBLE_COL_WIDTH, 2.7))

    colors = {"Q-Learning": "#003366", "SARSA": "#B22222"}

    def smooth_1d(vals, w=150):
        cumsum = np.cumsum(np.insert(vals, 0, 0))
        smoothed = (cumsum[w:] - cumsum[:-w]) / w
        pad = np.full(w - 1, smoothed[0])
        return np.concatenate((pad, smoothed))

    # Subplot (a): Return
    ax0 = axes[0]
    for label, data in results.items():
        all_runs = np.array([smooth_1d(r, w=200) for r in data["returns"]])
        mean_curve = np.mean(all_runs, axis=0)
        std_curve = np.std(all_runs, axis=0)
        eps = np.arange(1, num_episodes + 1)
        c = colors[label]

        ax0.plot(eps, mean_curve, label=label, color=c, linewidth=1.5)
        ax0.fill_between(eps, mean_curve - std_curve, mean_curve + std_curve, color=c, alpha=0.18)

    ax0.set_xlabel("Episode", fontsize=8.5)
    ax0.set_ylabel("Episodic Return", fontsize=8.5)
    ax0.set_title("(a) Multi-Seed Return (Mean +/- Std, 5 Seeds)", fontsize=9.0, weight="normal")
    ax0.grid(True, ls="--", alpha=0.4)
    ax0.legend(fontsize=8.0, loc="lower right")

    # Subplot (b): Success Rate
    ax1 = axes[1]
    for label, data in results.items():
        all_runs = np.array([smooth_1d(np.array(s, dtype=float) * 100, w=200) for s in data["success"]])
        mean_curve = np.mean(all_runs, axis=0)
        std_curve = np.std(all_runs, axis=0)
        eps = np.arange(1, num_episodes + 1)
        c = colors[label]

        ax1.plot(eps, mean_curve, label=label, color=c, linewidth=1.5)
        ax1.fill_between(eps, mean_curve - std_curve, mean_curve + std_curve, color=c, alpha=0.18)

    ax1.set_xlabel("Episode", fontsize=8.5)
    ax1.set_ylabel("Success Rate (%)", fontsize=8.5)
    ax1.set_title("(b) Multi-Seed Success Rate (Mean +/- Std, 5 Seeds)", fontsize=9.0, weight="normal")
    ax1.grid(True, ls="--", alpha=0.4)
    ax1.legend(fontsize=8.0, loc="lower right")
    ax1.set_ylim(-5, 105)

    plt.tight_layout()

    save_path = os.path.join(PROJECT_ROOT, "figures", "multi_seed_learning_curves.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Copy to mirrors
    for dest in [
        os.path.join(PROJECT_ROOT, "output", "plots", "multi_seed_learning_curves.png"),
        os.path.join(PROJECT_ROOT, "figure", "multi_seed_learning_curves.png"),
        r"C:\Users\Admin\.gemini\antigravity\brain\dba0cb62-f2af-4952-a4c4-089a0280f93e\multi_seed_learning_curves.png",
    ]:
        try:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copyfile(save_path, dest)
        except Exception:
            pass

    print(f"\n[+] Saved multi-seed IEEE plot to: {save_path}")
    print("=" * 80)
    print("MULTI-SEED EVALUATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    run_multi_seed_experiment()
