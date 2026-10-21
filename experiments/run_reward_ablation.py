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
    print("REWARD CONFIGURATION ABLATION STUDY (DILEMMA MAP)")
    print("=" * 80)

    map_path = os.path.join(PROJECT_ROOT, "maps", "map_7x7_dilemma.txt")

    configs = {
        "Standard (R_step=-1, R_dl=-20, R_pull=-0.5)": {
            "step_penalty": -1.0,
            "goal_reward": 100.0,
            "deadlock_penalty": -20.0,
            "pull_cost": -0.5,
            "color": "#003366",
            "ls": "-",
            "category": "Optimal Baseline",
        },
        "Zero Step Penalty (R_step=0, R_dl=-20, R_pull=-0.5)": {
            "step_penalty": 0.0,
            "goal_reward": 100.0,
            "deadlock_penalty": -20.0,
            "pull_cost": -0.5,
            "color": "#D95F02",
            "ls": "--",
            "category": "Degenerate (No Urgency)",
        },
        "Severe Deadlock (R_step=-1, R_dl=-60, R_pull=-0.5)": {
            "step_penalty": -1.0,
            "goal_reward": 100.0,
            "deadlock_penalty": -60.0,
            "pull_cost": -0.5,
            "color": "#B22222",
            "ls": "-.",
            "category": "Overly Risk-Averse",
        },
        "Heavy Pull Cost (R_step=-1, R_dl=-20, R_pull=-15.0)": {
            "step_penalty": -1.0,
            "goal_reward": 100.0,
            "deadlock_penalty": -20.0,
            "pull_cost": -15.0,
            "color": "#1B4D3E",
            "ls": ":",
            "category": "Pull-Frugal Detour",
        },
    }

    # Pre-extract transition dynamics once (kinematics are invariant to reward)
    print("\n1. Building baseline transition model (P matrix invariant to reward)...")
    base_env = SokobanPullEnv(map_path, mode="special")
    P_shared, _ = base_env.get_transition_model()
    num_states = base_env.num_states
    print(f"   [+] State space size: |S| = {num_states}, Action space size: |A| = {base_env.num_actions}")

    dp_results = {}
    rollout_results = {}

    print("\n2. Executing Value Iteration across reward configurations:")
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

        # Rollout optimal policy
        obs, info = env.reset()
        traj_steps = 0
        traj_ret = 0.0
        traj_pulls = 0
        done = False
        while not done and traj_steps < 100:
            a = vi["policy"][info["state_idx"]]
            if a == 4:
                traj_pulls += 1
            obs, r, term, trunc, info = env.step(a)
            traj_steps += 1
            traj_ret += r
            done = term or trunc

        rollout_results[name] = {
            "steps": traj_steps,
            "pulls": traj_pulls,
            "return": traj_ret,
            "solved": env.is_solved,
        }

        print(f"   [+] {name}")
        print(f"       VI Iterations: {vi['num_iterations']:3d} | V* Range: [{v_min:6.1f}, {v_max:6.1f}]")
        print(f"       Trajectory:    Steps={traj_steps} | Pulls={traj_pulls} | Return={traj_ret:.1f} | Solved={env.is_solved}")

    # -------------------------------------------------------------------------
    # Policy Divergence Analysis
    # -------------------------------------------------------------------------
    std_key = "Standard (R_step=-1, R_dl=-20, R_pull=-0.5)"
    pi_std = dp_results[std_key]["policy"]

    print("\n" + "=" * 80)
    print("POLICY DIVERGENCE & ACTION DISTRIBUTION ANALYSIS")
    print("=" * 80)
    print(f"{'Configuration':<42} | {'Diff States':<11} | {'Diff %':<7} | {'PULL States':<11} | {'Steps':<6} | {'Pulls':<6} | {'Assessment'}")
    print("-" * 115)

    summary_rows = []
    for name, cfg in configs.items():
        pi_curr = dp_results[name]["policy"]
        diff_count = int(np.sum(pi_curr != pi_std))
        diff_pct = (diff_count / num_states) * 100.0
        pull_states = int(np.sum(pi_curr == 4))

        ro = rollout_results[name]
        vi_iters = int(dp_results[name]["num_iterations"])

        print(f"{name:<42} | {diff_count:11d} | {diff_pct:6.1f}% | {pull_states:11d} | {ro['steps']:6d} | {ro['pulls']:6d} | {cfg['category']}")

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
            "trajectory_steps": ro["steps"],
            "trajectory_pulls": ro["pulls"],
            "trajectory_return": round(ro["return"], 2),
            "solved": ro["solved"],
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

    cfg_labels = ["Standard", "Zero Step", "Severe DL", "Heavy Pull"]
    cfg_names = list(configs.keys())

    # Panel (a): Policy Alteration & Rescue Pull States
    ax0 = axes[0]
    x = np.arange(len(cfg_labels))
    width = 0.35
    pull_counts = [int(np.sum(dp_results[k]["policy"] == 4)) for k in cfg_names]
    diff_pcts = [(int(np.sum(dp_results[k]["policy"] != pi_std)) / num_states) * 100.0 for k in cfg_names]

    b1 = ax0.bar(x - width/2, pull_counts, width, label="PULL States (|S_pull|)", color="#003366")
    ax0_twin = ax0.twinx()
    b2 = ax0_twin.bar(x + width/2, diff_pcts, width, label="Policy Diff (%)", color="#B22222", alpha=0.85)

    ax0.set_xticks(x)
    ax0.set_xticklabels(cfg_labels, fontsize=7.5)
    ax0.set_ylabel("States choosing PULL", fontsize=8.0, color="#003366")
    ax0_twin.set_ylabel("Policy Divergence (%)", fontsize=8.0, color="#B22222")
    ax0.set_title("(a) Policy Shift & Pull Allocation", fontsize=8.5, weight="normal")
    ax0.grid(True, ls="--", alpha=0.3, axis="y")
    ax0.set_ylim(0, 36)
    ax0_twin.set_ylim(0, 6)

    # Panel (b): Trajectory Navigation & Detour Trade-off
    ax1 = axes[1]
    traj_steps = [rollout_results[k]["steps"] for k in cfg_names]
    traj_pulls = [rollout_results[k]["pulls"] for k in cfg_names]

    b3 = ax1.bar(x - width/2, traj_steps, width, label="Optimal Steps", color="#1B4D3E")
    ax1_twin = ax1.twinx()
    b4 = ax1_twin.bar(x + width/2, traj_pulls, width, label="Rescue Pulls Used", color="#D95F02", alpha=0.85)

    ax1.set_xticks(x)
    ax1.set_xticklabels(cfg_labels, fontsize=7.5)
    ax1.set_ylabel("Steps to Goal", fontsize=8.0, color="#1B4D3E")
    ax1_twin.set_ylabel("Pulls Executed", fontsize=8.0, color="#D95F02")
    ax1.set_title("(b) Navigation & Detour Trade-off", fontsize=8.5, weight="normal")
    ax1.grid(True, ls="--", alpha=0.3, axis="y")
    ax1.set_ylim(0, 45)
    ax1_twin.set_ylim(0, 3)

    # Panel (c): VI Bellman Residual Decay
    ax2 = axes[2]
    for name, cfg in configs.items():
        errs = dp_results[name]["errors"]
        short_lbl = name.split(" (")[0]
        ax2.plot(errs, label=short_lbl, color=cfg["color"], linestyle=cfg["ls"], linewidth=1.3)
    ax2.set_yscale("log")
    ax2.set_xlabel("Iteration", fontsize=8.0)
    ax2.set_ylabel(r"Bellman Error $\|V_{k+1} - V_k\|_\infty$", fontsize=8.0)
    ax2.set_title("(c) DP Contraction Stability", fontsize=8.5, weight="normal")
    ax2.grid(True, which="both", ls="--", alpha=0.3)
    ax2.legend(fontsize=7.0, loc="upper right")

    plt.tight_layout()

    # Save to figures/ and output/
    save_path = os.path.join(PROJECT_ROOT, "figures", "ablation_reward_configurations.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Mirror copy
    dest_out = os.path.join(PROJECT_ROOT, "output", "plots", "ablation_reward_configurations.png")
    os.makedirs(os.path.dirname(dest_out), exist_ok=True)
    shutil.copyfile(save_path, dest_out)

    print(f"[+] Saved high-resolution IEEE plot to: {save_path}")
    print("=" * 80)
    print("REWARD ABLATION STUDY COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    run_reward_ablation()
