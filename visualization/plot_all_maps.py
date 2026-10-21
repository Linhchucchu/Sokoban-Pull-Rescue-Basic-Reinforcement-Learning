"""
Generates an IEEE publication-quality overview figure displaying all project maps:
Row 1: 5 Core Benchmark Settings (Setting 1 5x5, Setting 3 6x6, Setting 2 7x7, Setting 4 7x7, Setting 5 7x7 Dilemma)
Row 2: 4 Diverse Topological Variants (Central Vault 7x7, Zigzag 7x7, Labyrinth 6x6, Four Rooms 5x5)
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sokoban_env.sokoban_pull_env import SokobanPullEnv
from visualization.ieee_style import apply_ieee_style, IEEE_DOUBLE_COL_WIDTH


def plot_all_maps():
    apply_ieee_style()
    print("=" * 80)
    print("GENERATING COMPREHENSIVE OVERVIEW FIGURE FOR ALL BENCHMARK & VARIANT MAPS")
    print("=" * 80)

    core_maps = [
        {"title": "Setting 1: Basic Map", "filename": "map_5x5_basic.txt", "mode": "basic", "desc": "5x5 inner (7x7)\n|S| = 600, |A| = 4\nPush-only baseline"},
        {"title": "Setting 3: Rescue Map", "filename": "map_6x6_rescue.txt", "mode": "special", "desc": "6x6 inner (8x8)\n|S| = 2,808, |A| = 5\n1 rescue pull"},
        {"title": "Setting 2: Dead Corners", "filename": "map_7x7_deadcorners.txt", "mode": "special", "desc": "7x7 inner (9x9)\n|S| = 5,328, |A| = 5\nCorner traps, 1 pull"},
        {"title": "Setting 4: Hook Recharge", "filename": "map_7x7_hook.txt", "mode": "special", "desc": "7x7 inner (9x9)\n|S| = 5,328, |A| = 5\nHook tile (H) recharges"},
        {"title": "Setting 5: Dilemma Map", "filename": "map_7x7_dilemma.txt", "mode": "special", "desc": "7x7 inner (9x9)\n|S| = 6,240, |A| = 5\nShortest path = Trap"},
    ]

    variant_maps = [
        {"title": "Variant: Central Vault", "filename": "map_7x7_vault.txt", "mode": "special", "desc": "7x7 inner (9x9)\n|S| = 5,928, |A| = 5\nGoal dead center (4,4)"},
        {"title": "Variant: Zigzag S-Curve", "filename": "map_7x7_zigzag.txt", "mode": "special", "desc": "7x7 inner (9x9)\n|S| = 5,328, |A| = 5\nHorizontal barriers"},
        {"title": "Variant: Labyrinth Maze", "filename": "map_6x6_labyrinth.txt", "mode": "special", "desc": "6x6 inner (8x8)\n|S| = 3,024, |A| = 5\nStart bottom, Goal top"},
        {"title": "Variant: Inverted 5x5", "filename": "map_5x5_inverted.txt", "mode": "basic", "desc": "5x5 inner (7x7)\n|S| = 600, |A| = 4\nOpen floor (no walls)"},
    ]

    fig, axes = plt.subplots(2, 5, figsize=(IEEE_DOUBLE_COL_WIDTH * 1.5, 7.2))

    # Row 1: Core Maps
    for idx, cfg in enumerate(core_maps):
        ax = axes[0, idx]
        _draw_single_map(ax, cfg)

    # Row 2: Diverse Variant Maps (col 0..3 for 4 maps, col 4 empty / legend)
    for idx, cfg in enumerate(variant_maps):
        ax = axes[1, idx]
        _draw_single_map(ax, cfg)

    # Empty slot on Row 2, Col 4 used for a clean Legend & Overview Card
    ax_card = axes[1, 4]
    ax_card.axis("off")
    legend_text = (
        "Legend & Markers:\n"
        "-------------------\n"
        "S : Agent Start Pos\n"
        "B : Cargo Crate Pos\n"
        "G : Goal Target Pad\n"
        "H : Hook Recharge Tile\n"
        "Dark Block : Solid Wall\n\n"
        "Key Innovations:\n"
        "- Varied Goal Centers\n"
        "- Bottom/Corner Starts\n"
        "- Serpentine Corridors\n"
        "- Fortified Vaults"
    )
    ax_card.text(
        0.1, 0.5, legend_text,
        fontsize=8.5, family="serif", verticalalignment="center",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#F8FAFC", edgecolor="#CBD5E1", alpha=0.95)
    )

    fig.suptitle("Topic 15: Benchmark Suite of 9 Diverse Map Topologies (5 Settings + 4 Geometric Variants)", fontsize=11.0, weight="normal", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    for folder in ["figures", os.path.join("output", "plots")]:
        os.makedirs(folder, exist_ok=True)
        out_path = os.path.join(folder, "maps_all_settings_overview.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")

    plt.close()
    print("   [+] Saved enhanced 9-map overview figure to: figures/ & output/plots/maps_all_settings_overview.png")


def _draw_single_map(ax, cfg):
    map_path = os.path.join(PROJECT_ROOT, "maps", cfg["filename"])
    env = SokobanPullEnv(map_path, mode=cfg["mode"])

    h, w = env.height, env.width
    img = np.full((h, w, 3), 0.95)
    img[env.walls] = [0.22, 0.22, 0.25]

    ax.imshow(img)

    for r in range(h):
        for c in range(w):
            if env.walls[r, c]:
                continue
            elif (r, c) == env.initial_agent_pos:
                ax.text(c, r, "S", ha="center", va="center", color="white", weight="bold", fontsize=9.0,
                        bbox=dict(boxstyle="square,pad=0.2", facecolor="#1E88E5", edgecolor="#1565C0", alpha=0.9))
            elif (r, c) == env.initial_box_pos:
                ax.text(c, r, "B", ha="center", va="center", color="black", weight="bold", fontsize=9.0,
                        bbox=dict(boxstyle="square,pad=0.2", facecolor="#FFA500", edgecolor="#D97706", alpha=0.9))
            elif (r, c) == env.goal_pos:
                ax.text(c, r, "G", ha="center", va="center", color="white", weight="bold", fontsize=9.0,
                        bbox=dict(boxstyle="square,pad=0.2", facecolor="#2E8B57", edgecolor="#1B5E20", alpha=0.9))
            elif hasattr(env, "hook_pos") and env.hook_pos is not None and (r, c) == env.hook_pos:
                ax.text(c, r, "H", ha="center", va="center", color="white", weight="bold", fontsize=9.0,
                        bbox=dict(boxstyle="square,pad=0.2", facecolor="#7B1FA2", edgecolor="#4A148C", alpha=0.9))

    ax.set_title(cfg["title"], fontsize=8.8, weight="normal", pad=6)
    ax.set_xlabel(cfg["desc"], fontsize=7.2, labelpad=6)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])


if __name__ == "__main__":
    plot_all_maps()
