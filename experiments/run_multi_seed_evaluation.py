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
import csv
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


def evaluate_greedy_policy(env, agent):
    """
    Evaluates the deterministic greedy policy (epsilon=0) from initial configuration.
    """
    obs, info = env.reset()
    steps, ret, pulls = 0, 0.0, 0
    done = False
    while not done and steps < 100:
        s_idx = info["state_idx"]
        a = int(np.argmax(agent.Q[s_idx]))
        if a == 4:
            pulls += 1
        obs, r, term, trunc, info = env.step(a)
        steps += 1
        ret += r
        done = term or trunc
    return env.is_solved, steps, ret, pulls


def run_multi_seed_experiment():
    apply_ieee_style()
    print("=" * 80)
    print("MULTI-SEED STATISTICAL EVALUATION (DILEMMA MAP: 5K, 10K, 30K EPISODES)")
    print("=" * 80)

    map_path = os.path.join(PROJECT_ROOT, "maps", "map_7x7_dilemma.txt")
    env = SokobanPullEnv(map_path, mode="special")

    seeds = [42, 100, 2024, 777, 999]
    checkpoints = [5000, 10000, 30000]
    algos = ["q_learning", "sarsa"]

    results = {
        "Q-Learning": {"returns": [], "success": [], "lengths": []},
        "SARSA": {"returns": [], "success": [], "lengths": []},
    }
    checkpoint_rows = []

    for algo in algos:
        label = "Q-Learning" if algo == "q_learning" else "SARSA"
        print(f"\n" + "-" * 70)
        print(f"Evaluating {label} across {len(seeds)} random seeds on Dilemma Map...")
        print("-" * 70)

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

            all_returns = []
            all_success = []
            all_lengths = []
            current_ep = 0

            for target_ep in checkpoints:
                ep_delta = target_ep - current_ep
                res = agent.train(num_episodes=ep_delta, log_interval=ep_delta)
                current_ep = target_ep

                all_returns.extend(res["returns"])
                all_success.extend(res["success"])
                all_lengths.extend(res["lengths"])

                # 1. Online Training Metrics (last 500 episodes under epsilon-greedy)
                online_succ = float(np.mean(res["success"][-500:]) * 100)
                online_ret = float(np.mean(res["returns"][-500:]))
                online_steps = float(np.mean(res["lengths"][-500:]))

                # 2. Greedy Evaluation (deterministic epsilon=0 from start)
                solved, g_steps, g_ret, g_pulls = evaluate_greedy_policy(env, agent)

                print(
                    f"[{label:10s}] Seed {seed:4d} @ {target_ep:5d} eps | "
                    f"Online: Succ={online_succ:5.1f}%, Ret={online_ret:6.1f} | "
                    f"Greedy: Solved={str(solved):<5}, Steps={g_steps:2d}, Ret={g_ret:5.1f}, Pulls={g_pulls}"
                )

                checkpoint_rows.append({
                    "algorithm": label,
                    "seed": seed,
                    "episodes": target_ep,
                    "online_success_rate_pct": round(online_succ, 2),
                    "online_mean_return": round(online_ret, 2),
                    "online_mean_steps": round(online_steps, 2),
                    "greedy_solved": solved,
                    "greedy_steps": g_steps,
                    "greedy_return": round(g_ret, 2),
                    "greedy_pulls": g_pulls,
                })

            results[label]["returns"].append(all_returns)
            results[label]["success"].append(all_success)
            results[label]["lengths"].append(all_lengths)

    # 1. Granular Seed-Level Checkpoints CSV
    csv_dilemma_out = os.path.join(PROJECT_ROOT, "output", "tables", "dilemma_multi_seed_evaluation.csv")
    csv_dilemma_data = os.path.join(PROJECT_ROOT, "data", "dilemma_multi_seed_evaluation.csv")
    for p in [csv_dilemma_out, csv_dilemma_data]:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as f:
            fieldnames = list(checkpoint_rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(checkpoint_rows)

    # 2. Aggregated Statistical Performance Summary CSV (Mean +/- Std per checkpoint)
    summary_rows = []
    for algo_lbl in ["Q-Learning", "SARSA"]:
        for ep_ckpt in checkpoints:
            sub = [r for r in checkpoint_rows if r["algorithm"] == algo_lbl and r["episodes"] == ep_ckpt]
            on_succ = [r["online_success_rate_pct"] for r in sub]
            on_ret = [r["online_mean_return"] for r in sub]
            gr_steps = [r["greedy_steps"] for r in sub]
            gr_ret = [r["greedy_return"] for r in sub]
            gr_pulls = [r["greedy_pulls"] for r in sub]
            gr_solved = [1 if r["greedy_solved"] else 0 for r in sub]

            summary_rows.append({
                "algorithm": algo_lbl,
                "episodes": ep_ckpt,
                "online_success_mean": round(float(np.mean(on_succ)), 2),
                "online_success_std": round(float(np.std(on_succ, ddof=1)), 2),
                "online_return_mean": round(float(np.mean(on_ret)), 2),
                "online_return_std": round(float(np.std(on_ret, ddof=1)), 2),
                "greedy_solved_pct": round(float(np.mean(gr_solved)) * 100, 1),
                "greedy_steps_mean": round(float(np.mean(gr_steps)), 1),
                "greedy_steps_std": round(float(np.std(gr_steps, ddof=1)), 1),
                "greedy_return_mean": round(float(np.mean(gr_ret)), 2),
                "greedy_return_std": round(float(np.std(gr_ret, ddof=1)), 2),
                "greedy_pulls_mean": round(float(np.mean(gr_pulls)), 1),
            })

    csv_summary_out = os.path.join(PROJECT_ROOT, "output", "tables", "multi_seed_performance_summary.csv")
    csv_summary_data = os.path.join(PROJECT_ROOT, "data", "multi_seed_performance_summary.csv")
    for p in [csv_summary_out, csv_summary_data]:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as f:
            fieldnames = list(summary_rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"\n[+] Exported detailed dilemma multi-seed checkpoint table to: {csv_dilemma_out}")
    print(f"[+] Exported aggregated multi-seed summary table to: {csv_summary_out}")

    # -------------------------------------------------------------------------
    # Plotting Mean +/- Std Error across Seeds up to 30,000 Episodes
    # -------------------------------------------------------------------------
    print("\nGenerating IEEE Plot: figures/multi_seed_learning_curves.png...")
    fig, axes = plt.subplots(1, 2, figsize=(IEEE_DOUBLE_COL_WIDTH, 2.7))

    colors = {"Q-Learning": "#003366", "SARSA": "#B22222"}

    def smooth_1d(vals, w=300):
        if len(vals) < w:
            return vals
        cumsum = np.cumsum(np.insert(vals, 0, 0))
        smoothed = (cumsum[w:] - cumsum[:-w]) / w
        pad = np.full(w - 1, smoothed[0])
        return np.concatenate((pad, smoothed))

    total_episodes = 30000

    # Subplot (a): Return
    ax0 = axes[0]
    for label, data in results.items():
        all_runs = np.array([smooth_1d(r, w=400) for r in data["returns"]])
        mean_curve = np.mean(all_runs, axis=0)
        std_curve = np.std(all_runs, axis=0)
        eps = np.arange(1, total_episodes + 1)
        c = colors[label]

        ax0.plot(eps, mean_curve, label=label, color=c, linewidth=1.5)
        ax0.fill_between(eps, mean_curve - std_curve, mean_curve + std_curve, color=c, alpha=0.18)

    ax0.set_xlabel("Episode", fontsize=8.5)
    ax0.set_ylabel("Episodic Return", fontsize=8.5)
    ax0.set_title("(a) Multi-Seed Return (Dilemma Map, 5 Seeds)", fontsize=9.0, weight="normal")
    ax0.grid(True, ls="--", alpha=0.4)
    ax0.legend(fontsize=8.0, loc="lower right")

    # Subplot (b): Success Rate
    ax1 = axes[1]
    for label, data in results.items():
        all_runs = np.array([smooth_1d(np.array(s, dtype=float) * 100, w=400) for s in data["success"]])
        mean_curve = np.mean(all_runs, axis=0)
        std_curve = np.std(all_runs, axis=0)
        eps = np.arange(1, total_episodes + 1)
        c = colors[label]

        ax1.plot(eps, mean_curve, label=label, color=c, linewidth=1.5)
        ax1.fill_between(eps, mean_curve - std_curve, mean_curve + std_curve, color=c, alpha=0.18)

    ax1.set_xlabel("Episode", fontsize=8.5)
    ax1.set_ylabel("Online Success Rate (%)", fontsize=8.5)
    ax1.set_title("(b) Multi-Seed Success Rate (Dilemma Map, 5 Seeds)", fontsize=9.0, weight="normal")
    ax1.grid(True, ls="--", alpha=0.4)
    ax1.legend(fontsize=8.0, loc="lower right")
    ax1.set_ylim(-5, 105)

    plt.tight_layout()

    save_path = os.path.join(PROJECT_ROOT, "figures", "multi_seed_learning_curves.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Copy to mirror
    dest_plot = os.path.join(PROJECT_ROOT, "output", "plots", "multi_seed_learning_curves.png")
    os.makedirs(os.path.dirname(dest_plot), exist_ok=True)
    shutil.copyfile(save_path, dest_plot)

    print(f"\n[+] Saved multi-seed IEEE plot to: {save_path}")
    print("=" * 80)
    print("MULTI-SEED DILEMMA EVALUATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    run_multi_seed_experiment()
