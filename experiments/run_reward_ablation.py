"""
Reward Configuration Ablation Study for Project 15 Sokoban.
Compares 4 distinct reward shaping configurations:
1. Standard Balanced: Goal=+100, Step=-1.0, Deadlock=-20, PullCost=-0.5 (Default baseline)
2. Zero Step Penalty: Goal=+100, Step=0.0, Deadlock=-20, PullCost=-0.5 (No time pressure, wanders)
3. Severe Deadlock Penalty: Goal=+100, Step=-1.0, Deadlock=-60, PullCost=-0.5 (Extreme risk-aversion)
4. Heavy Pull Cost: Goal=+100, Step=-1.0, Deadlock=-20, PullCost=-5.0 (Expensive pull rescue)

Generates IEEE-format publication plot: figures/ablation_reward_configurations.png
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
from algorithms.dynamic_programming import DynamicProgramming
from algorithms.temporal_difference import TemporalDifferenceControl
from visualization.ieee_style import apply_ieee_style, IEEE_DOUBLE_COL_WIDTH


def run_reward_ablation():
    apply_ieee_style()
    print("=" * 80)
    print("REWARD CONFIGURATION ABLATION STUDY")
    print("=" * 80)

    map_path = os.path.join(PROJECT_ROOT, "maps", "map_6x6_rescue.txt")

    configs = {
        "Standard (R_step=-1, R_dl=-20)": {
            "step_penalty": -1.0,
            "goal_reward": 100.0,
            "deadlock_penalty": -20.0,
            "pull_cost": -0.5,
            "color": "#003366",
            "ls": "-",
            "category": "Optimal Baseline",
        },
        "Zero Step Penalty (R_step=0)": {
            "step_penalty": 0.0,
            "goal_reward": 100.0,
            "deadlock_penalty": -20.0,
            "pull_cost": -0.5,
            "color": "#D95F02",
            "ls": "--",
            "category": "Degenerate (No Urgency)",
        },
        "Severe Deadlock (R_dl=-60)": {
            "step_penalty": -1.0,
            "goal_reward": 100.0,
            "deadlock_penalty": -60.0,
            "pull_cost": -0.5,
            "color": "#B22222",
            "ls": "-.",
            "category": "Overly Risk-Averse",
        },
        "Heavy Pull Cost (R_pull=-15.0)": {
            "step_penalty": -1.0,
            "goal_reward": 100.0,
            "deadlock_penalty": -20.0,
            "pull_cost": -15.0,
            "color": "#1B4D3E",
            "ls": ":",
            "category": "Rescue-Frugal Detour",
        },
    }

    # Pre-extract transition dynamics once (kinematics are invariant to reward)
    print("\n1. Building baseline transition model (P matrix invariant to reward)...")
    base_env = SokobanPullEnv(map_path, mode="special")
    P_shared, _ = base_env.get_transition_model()
    num_states = base_env.num_states
    print(f"   [+] State space size: |S| = {num_states}, Action space size: |A| = {base_env.num_actions}")

    num_episodes = 6000
    td_results = {}
    dp_results = {}

    print("\n2. Executing Value Iteration and Q-Learning across reward configurations:")
    for name, cfg in configs.items():
        env = SokobanPullEnv(
            map_path,
            mode="special",
            step_penalty=cfg["step_penalty"],
            goal_reward=cfg["goal_reward"],
            deadlock_penalty=cfg["deadlock_penalty"],
            pull_cost=cfg["pull_cost"],
        )

        # Build fast reward matrix using env's specific reward formulation
        R = np.zeros((num_states, env.num_actions), dtype=np.float64)
        for s_idx, state in enumerate(env.all_states):
            for a in range(env.num_actions):
                _, r, _, _, _ = env.simulate_step(state, a)
                R[s_idx, a] = r

        # Dynamic Programming Value Iteration
        dp = DynamicProgramming(P_shared, R, gamma=0.9)
        vi = dp.value_iteration(theta=1e-7)
        dp_results[name] = vi
        v_min, v_max = vi["V"].min(), vi["V"].max()

        # Q-Learning Model-Free
        agent = TemporalDifferenceControl(
            env,
            algo="q_learning",
            gamma=0.9,
            alpha=0.1,
            epsilon_start=1.0,
            epsilon_min=0.05,
            epsilon_decay=0.9995,
        )
        res = agent.train(num_episodes=num_episodes, log_interval=2000)
        td_results[name] = res

        final_succ = np.mean(res["success"][-500:]) * 100
        final_steps = np.mean(res["lengths"][-500:])
        final_ret = np.mean(res["returns"][-500:])

        print(f"   [+] {name}")
        print(f"       VI Iterations: {vi['num_iterations']:3d} | V* Range: [{v_min:6.1f}, {v_max:6.1f}]")
        print(f"       Success Rate:  {final_succ:5.1f}% | Mean Steps: {final_steps:5.1f} | Mean Return: {final_ret:6.1f}")

    # -------------------------------------------------------------------------
    # Policy Divergence Analysis
    # -------------------------------------------------------------------------
    std_key = "Standard (R_step=-1, R_dl=-20)"
    pi_std = dp_results[std_key]["policy"]

    print("\n" + "=" * 80)
    print("POLICY DIVERGENCE & ACTION DISTRIBUTION ANALYSIS")
    print("=" * 80)
    print(f"{'Configuration':<38} | {'Diff States':<11} | {'Diff %':<7} | {'PULL States':<11} | {'Assessment'}")
    print("-" * 105)

    summary_rows = []
    for name, cfg in configs.items():
        pi_curr = dp_results[name]["policy"]
        diff_count = int(np.sum(pi_curr != pi_std))
        diff_pct = (diff_count / num_states) * 100.0
        pull_states = int(np.sum(pi_curr == 4))

        final_succ = float(np.mean(td_results[name]["success"][-500:]) * 100)
        final_steps = float(np.mean(td_results[name]["lengths"][-500:]))
        final_ret = float(np.mean(td_results[name]["returns"][-500:]))
        vi_iters = int(dp_results[name]["num_iterations"])

        print(f"{name:<38} | {diff_count:11d} | {diff_pct:6.1f}% | {pull_states:11d} | {cfg['category']}")

        summary_rows.append({
            "configuration": name,
            "category": cfg["category"],
            "step_penalty": cfg["step_penalty"],
            "goal_reward": cfg["goal_reward"],
            "deadlock_penalty": cfg["deadlock_penalty"],
            "pull_cost": cfg["pull_cost"],
            "vi_iterations": vi_iters,
            "policy_diff_states": diff_count,
            "policy_diff_pct": round(diff_pct, 2),
            "pull_action_states": pull_states,
            "success_rate_pct": round(final_succ, 2),
            "mean_steps": round(final_steps, 2),
            "mean_return": round(final_ret, 2),
        })

    # Save summary table to CSV
    csv_path_out = os.path.join(PROJECT_ROOT, "output", "tables", "reward_ablation_policy_comparison.csv")
    csv_path_data = os.path.join(PROJECT_ROOT, "data", "reward_ablation_policy_comparison.csv")
    os.makedirs(os.path.dirname(csv_path_out), exist_ok=True)
    os.makedirs(os.path.dirname(csv_path_data), exist_ok=True)

    fieldnames = list(summary_rows[0].keys())
    for target_csv in [csv_path_out, csv_path_data]:
        with open(target_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
    print(f"\n[+] Exported quantitative policy comparison table to: {csv_path_out}")

    # -------------------------------------------------------------------------
    # Plotting IEEE-Style Multi-Panel Figure
    # -------------------------------------------------------------------------
    print("\n2. Generating IEEE Publication Plot: ablation_reward_configurations.png...")
    fig, axes = plt.subplots(1, 3, figsize=(IEEE_DOUBLE_COL_WIDTH, 2.7))

    def smooth(vals, w=200):
        if len(vals) < w:
            return vals
        cumsum = np.cumsum(np.insert(vals, 0, 0))
        smoothed = (cumsum[w:] - cumsum[:-w]) / w
        pad = np.full(w - 1, smoothed[0])
        return np.concatenate((pad, smoothed))

    # Panel (a): Rolling Success Rate (%)
    ax0 = axes[0]
    for name, cfg in configs.items():
        res = td_results[name]
        succ_rate = smooth(np.array(res["success"], dtype=float) * 100, w=300)
        ax0.plot(succ_rate, label=name, color=cfg["color"], linestyle=cfg["ls"], linewidth=1.3)
    ax0.set_xlabel("Episode", fontsize=8.5)
    ax0.set_ylabel("Success Rate (%)", fontsize=8.5)
    ax0.set_title("(a) Convergence Rate", fontsize=9.0, weight="normal")
    ax0.grid(True, ls="--", alpha=0.4)
    ax0.set_ylim(-5, 105)

    # Panel (b): Steps to Goal
    ax1 = axes[1]
    for name, cfg in configs.items():
        res = td_results[name]
        steps_smooth = smooth(res["lengths"], w=300)
        ax1.plot(steps_smooth, label=name, color=cfg["color"], linestyle=cfg["ls"], linewidth=1.3)
    ax1.set_xlabel("Episode", fontsize=8.5)
    ax1.set_ylabel("Steps to Goal", fontsize=8.5)
    ax1.set_title("(b) Navigation Efficiency", fontsize=9.0, weight="normal")
    ax1.grid(True, ls="--", alpha=0.4)

    # Panel (c): VI Bellman Residual Decay
    ax2 = axes[2]
    for name, cfg in configs.items():
        errs = dp_results[name]["errors"]
        ax2.plot(errs, label=name, color=cfg["color"], linestyle=cfg["ls"], linewidth=1.3)
    ax2.set_yscale("log")
    ax2.set_xlabel("Iteration", fontsize=8.5)
    ax2.set_ylabel(r"Bellman Error $\|V_{k+1} - V_k\|_\infty$", fontsize=8.5)
    ax2.set_title("(c) DP Contraction Stability", fontsize=9.0, weight="normal")
    ax2.grid(True, which="both", ls="--", alpha=0.4)

    # Global legend at bottom
    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        fontsize=7.5,
        framealpha=0.9,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)

    # Save to figures/ and output/
    save_path = os.path.join(PROJECT_ROOT, "figures", "ablation_reward_configurations.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Mirror copy
    dest_out = os.path.join(PROJECT_ROOT, "output", "plots", "ablation_reward_configurations.png")
    dest_fig = os.path.join(PROJECT_ROOT, "figure", "ablation_reward_configurations.png")
    dest_brain = r"C:\Users\Admin\.gemini\antigravity\brain\dba0cb62-f2af-4952-a4c4-089a0280f93e\ablation_reward_configurations.png"
    for d in [dest_out, dest_fig, dest_brain]:
        try:
            os.makedirs(os.path.dirname(d), exist_ok=True)
            shutil.copyfile(save_path, d)
        except Exception:
            pass

    print(f"[+] Saved high-resolution IEEE plot to: {save_path}")
    print("=" * 80)
    print("REWARD ABLATION STUDY COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    run_reward_ablation()
