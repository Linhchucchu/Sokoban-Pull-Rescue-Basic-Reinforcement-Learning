import os
import sys
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sokoban_env.sokoban_pull_env import SokobanPullEnv
from algorithms.dynamic_programming import DynamicProgramming
from visualization.ieee_style import apply_ieee_style, IEEE_DOUBLE_COL_WIDTH

def generate_setting4_policy_visualization():
    apply_ieee_style()
    print("Generating Setting 4 Policy Slice and Trajectory Trace...")

    map_path = os.path.join(PROJECT_ROOT, "maps", "map_7x7_dilemma.txt")
    env = SokobanPullEnv(map_path, mode="special")
    P, R = env.get_transition_model()
    dp = DynamicProgramming(P, R, gamma=0.9)
    vi = dp.value_iteration(theta=1e-7)
    V = vi["V"]
    policy = vi["policy"]

    # 1. Simulate 31-step optimal trajectory
    obs, info = env.reset(seed=42)
    agent_path = [env.agent_pos]
    box_path = [env.box_pos]
    actions_taken = []
    events = []

    done = False
    step_count = 0
    while not done and step_count < 100:
        s = info["state_idx"]
        a = policy[s]
        actions_taken.append(a)
        obs, r, term, trunc, info = env.step(a)
        step_count += 1
        agent_path.append(env.agent_pos)
        box_path.append(env.box_pos)
        events.append(info.get("event", "move"))
        done = term or trunc

    # 2. Extract 2D Slice with Box fixed at initial position (4, 2), pull_avail=1, hook_used=0
    fixed_box = env.initial_box_pos  # (4, 2)
    grid_V = np.full((env.height, env.width), np.nan, dtype=np.float64)
    grid_pi = np.full((env.height, env.width), -1, dtype=int)

    for ar, ac in env.free_cells:
        if (ar, ac) == fixed_box:
            continue
        st = (ar, ac, fixed_box[0], fixed_box[1], 1, 0)
        if st in env.state_to_idx_dict:
            idx = env.state_to_idx_dict[st]
            grid_V[ar, ac] = V[idx]
            grid_pi[ar, ac] = policy[idx]

    # Create 2-panel figure
    fig, axes = plt.subplots(1, 2, figsize=(IEEE_DOUBLE_COL_WIDTH, 4.2), dpi=300)

    # -------------------------------------------------------------
    # Panel (a): 2D Policy & Value Slice (Box fixed at (4,2))
    # -------------------------------------------------------------
    ax1 = axes[0]
    ax1.set_title(r"(a) Policy Vector Field $\pi^*(s)$ & Value $V^*(s)$" "\n" r"(Box fixed at $(4,2)$, $\text{pull}=1$, $\text{hook}=0$)", fontsize=9, pad=6)

    # Base grid
    wall_mask = env.walls.copy()
    display_grid = np.where(wall_mask, np.nan, grid_V)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#222222")

    im = ax1.imshow(display_grid, cmap=cmap, origin="upper")
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label(r"State Value $V^*(s)$", fontsize=8)

    # Draw grid lines
    ax1.set_xticks(np.arange(-0.5, env.width, 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, env.height, 1), minor=True)
    ax1.grid(which="minor", color="#444444", linestyle="-", linewidth=0.5)
    ax1.tick_params(which="both", size=0, labelsize=7)

    # Mark Goal, Hook, Box
    gr, gc = env.goal_pos
    hr, hc = env.hook_pos
    br, bc = fixed_box

    ax1.text(gc, gr, "G", color="red", ha="center", va="center", fontweight="bold", fontsize=10, bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="red", lw=1))
    ax1.text(hc, hr, "H", color="blue", ha="center", va="center", fontweight="bold", fontsize=10, bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="blue", lw=1))
    ax1.text(bc, br, "B", color="#D95F02", ha="center", va="center", fontweight="bold", fontsize=10, bbox=dict(boxstyle="circle,pad=0.2", fc="#FFD92F", ec="#D95F02", lw=1.5))

    # Action arrows
    # 0: UP (dr=-1), 1: DOWN (dr=+1), 2: LEFT (dc=-1), 3: RIGHT (dc=+1), 4: PULL
    action_dx_dy = {
        0: (0, -0.32),
        1: (0, 0.32),
        2: (-0.32, 0),
        3: (0.32, 0)
    }

    for r in range(env.height):
        for c in range(env.width):
            if env.walls[r, c] or (r, c) == fixed_box:
                continue
            act = grid_pi[r, c]
            if act in action_dx_dy:
                dx, dy = action_dx_dy[act]
                ax1.arrow(c, r, dx, dy, head_width=0.18, head_length=0.18, fc="white", ec="white", length_includes_head=True, lw=1.0)
            elif act == 4:
                # PULL
                ax1.plot(c, r, "o", color="#E7298A", markersize=6)

    # -------------------------------------------------------------
    # Panel (b): Optimal 31-Step Trajectory Trace
    # -------------------------------------------------------------
    ax2 = axes[1]
    ax2.set_title(r"(b) Coordinated 31-Step Optimal Trajectory Trace" "\n" r"(2 Rescue Pulls, 1 Hook Recharge, Return $+69.00$)", fontsize=9, pad=6)

    # Wall background
    bg = np.where(wall_mask, 0.2, 0.95)
    ax2.imshow(bg, cmap="gray", vmin=0, vmax=1, origin="upper")
    ax2.set_xticks(np.arange(-0.5, env.width, 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, env.height, 1), minor=True)
    ax2.grid(which="minor", color="#cccccc", linestyle="-", linewidth=0.5)
    ax2.tick_params(which="both", size=0, labelsize=7)

    # Static elements
    ax2.text(gc, gr, "G", color="red", ha="center", va="center", fontweight="bold", fontsize=10, bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="red", lw=1.2))
    ax2.text(hc, hr, "H", color="blue", ha="center", va="center", fontweight="bold", fontsize=10, bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="blue", lw=1.2))
    sr, sc = env.initial_agent_pos
    ax2.text(sc, sr, "S", color="#1B9E77", ha="center", va="center", fontweight="bold", fontsize=10, bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="#1B9E77", lw=1.2))

    # Trace paths
    ap = np.array(agent_path)
    bp = np.array(box_path)

    # Plot lines with jitter to avoid complete overlap
    ax2.plot(ap[:, 1], ap[:, 0], color="#1F78B4", lw=2, linestyle="-", label="Agent Path (31 steps)", alpha=0.85)
    ax2.plot(bp[:, 1] + 0.06, bp[:, 0] + 0.06, color="#FF7F00", lw=2.2, linestyle="--", label="Box Path (Towed/Pushed)", alpha=0.9)

    # Mark key milestones
    # Step 1: Start
    ax2.scatter(sc, sr, color="#1B9E77", s=50, zorder=5)
    # Pull 1 location
    pull_indices = [i for i, a in enumerate(actions_taken) if a == 4]
    for p_num, p_idx in enumerate(pull_indices):
        pr, pc = agent_path[p_idx]
        ax2.scatter(pc, pr, color="#E7298A", s=80, marker="*", zorder=6, label="Rescue Pull" if p_num == 0 else "")
        ax2.text(pc + 0.25, pr - 0.25, f"Pull {p_num+1}\n(t={p_idx+1})", color="#E7298A", fontsize=6.5, fontweight="bold")

    ax2.legend(loc="lower left", fontsize=6.8, framealpha=0.9)

    plt.tight_layout()
    out_dir = os.path.join(PROJECT_ROOT, "figures")
    out_plot = os.path.join(out_dir, "policy_trajectory_dilemma_setting4.png")
    plt.savefig(out_plot, dpi=300)
    
    # Mirror in output/plots
    out_mirror = os.path.join(PROJECT_ROOT, "output", "plots", "policy_trajectory_dilemma_setting4.png")
    os.makedirs(os.path.dirname(out_mirror), exist_ok=True)
    plt.savefig(out_mirror, dpi=300)
    plt.close()

    print(f"Successfully generated Setting 4 policy & trajectory plot: {out_plot}")

if __name__ == "__main__":
    generate_setting4_policy_visualization()
