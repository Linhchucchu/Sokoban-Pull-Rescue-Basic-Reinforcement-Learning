"""
David Silver RL Course style visualizer.
Generates Gridworld Value Function Heatmaps and Greedy Policy Arrow fields (Lecture 3 - Slide 11).
"""

from typing import List, Tuple, Dict, Any, Optional
import os
import matplotlib.pyplot as plt
import numpy as np

from visualization.ieee_style import apply_ieee_style, IEEE_DOUBLE_COL_WIDTH


class DavidSilverVisualizer:
    ACTION_ARROWS = {
        0: (0, -0.35),   # UP in grid coordinates (dr = -1) -> arrow dy = -0.35
        1: (0, 0.35),    # DOWN in grid coordinates (dr = +1) -> arrow dy = +0.35
        2: (-0.35, 0),   # LEFT (dc = -1) -> arrow dx = -0.35
        3: (0.35, 0),    # RIGHT (dc = +1) -> arrow dx = +0.35
        4: (0, 0),       # PULL -> circle or dot
    }
    ACTION_CHARS = {0: "↑", 1: "↓", 2: "←", 3: "→", 4: "●"}

    def __init__(self, env):
        self.env = env
        self.height = env.height
        self.width = env.width
        self.walls = env.walls
        self.goal_pos = env.goal_pos

    def extract_slice_2d(
        self,
        V: np.ndarray,
        policy: np.ndarray,
        box_pos: Tuple[int, int],
        pull_avail: int = 1,
        hook_used: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts 2D grid of values and actions for agent coordinates (r, c)
        given a fixed box position and pull/hook flags.
        """
        grid_V = np.full((self.height, self.width), np.nan, dtype=np.float64)
        grid_pi = np.full((self.height, self.width), -1, dtype=int)

        for ar, ac in self.env.free_cells:
            if (ar, ac) == box_pos:
                continue
            if self.env.mode == "basic":
                st = (ar, ac, box_pos[0], box_pos[1])
            else:
                st = (ar, ac, box_pos[0], box_pos[1], pull_avail, hook_used)

            if st in self.env.state_to_idx_dict:
                idx = self.env.state_to_idx_dict[st]
                grid_V[ar, ac] = V[idx]
                grid_pi[ar, ac] = policy[idx]

        return grid_V, grid_pi

    def plot_david_silver_iterations(
        self,
        history_V: List[np.ndarray],
        history_pi: List[np.ndarray],
        box_pos: Tuple[int, int],
        iterations_to_plot: List[int],
        save_path: str,
        title: str = "Iterative Policy Evaluation & Value Iteration in Sokoban (David Silver Lecture 3)",
    ):
        """
        Generates comparison subplots for iterations k = 0, 1, 2, 3, 10, converged.
        For each iteration, plots:
          1. Value Function Heatmap V_k(s)
          2. Greedy Policy Arrow Field pi_k(s)
        """
        n_iters = len(iterations_to_plot)
        apply_ieee_style()
        fig, axes = plt.subplots(2, n_iters, figsize=(max(IEEE_DOUBLE_COL_WIDTH, 2.2 * n_iters), 4.8))
        if n_iters == 1:
            axes = np.array([[axes[0]], [axes[1]]])

        # Get global min and max value for unified color scale
        valid_vals = []
        for col, it in enumerate(iterations_to_plot):
            idx = col if len(history_V) == len(iterations_to_plot) else min(it, len(history_V) - 1)
            gV, _ = self.extract_slice_2d(history_V[idx], history_pi[idx], box_pos)
            valid_vals.extend(gV[~np.isnan(gV)])
        vmin = min(valid_vals) if valid_vals else -10
        vmax = max(valid_vals) if valid_vals else 100

        for col, it in enumerate(iterations_to_plot):
            idx = col if len(history_V) == len(iterations_to_plot) else min(it, len(history_V) - 1)
            V_curr = history_V[idx]
            pi_curr = history_pi[idx]
            grid_V, grid_pi = self.extract_slice_2d(V_curr, pi_curr, box_pos)

            # Row 0: Value Heatmap
            ax_v = axes[0, col]
            # Draw floors as light neutral background and walls as solid charcoal blocks (no '#' text)
            base_img = np.full((self.height, self.width, 3), 0.94)
            base_img[self.walls] = [0.22, 0.22, 0.25]
            ax_v.imshow(base_img)

            masked_V = np.ma.masked_invalid(grid_V)
            im = ax_v.imshow(masked_V, cmap="Spectral", vmin=vmin, vmax=vmax, alpha=0.9)
            
            # Print numbers inside cells
            for r in range(self.height):
                for c in range(self.width):
                    if self.walls[r, c]:
                        pass  # Solid clean wall tile without '#' character
                    elif (r, c) == box_pos:
                        ax_v.text(c, r, "B", ha="center", va="center", color="black", weight="bold", fontsize=10, bbox=dict(boxstyle="square,pad=0.2", facecolor="#FFA500", edgecolor="#333333", alpha=0.9))
                    elif (r, c) == self.goal_pos:
                        ax_v.text(c, r, "G", ha="center", va="center", color="white", weight="bold", fontsize=10, bbox=dict(boxstyle="square,pad=0.2", facecolor="#2E8B57", edgecolor="#333333", alpha=0.9))
                    elif hasattr(self.env, "hook_tile_pos") and (r, c) == self.env.hook_tile_pos:
                        ax_v.text(c, r, "H", ha="center", va="center", color="white", weight="bold", fontsize=10, bbox=dict(boxstyle="square,pad=0.2", facecolor="#7B1FA2", edgecolor="#333333", alpha=0.9))
                    elif not np.isnan(grid_V[r, c]):
                        ax_v.text(c, r, f"{grid_V[r, c]:.1f}", ha="center", va="center", color="black", fontsize=8.0, weight="bold")

            ax_v.set_title(f"k = {it}\nValue Function V_k(s)", fontsize=9.5, weight="normal")
            ax_v.grid(False)
            ax_v.set_xticks([])
            ax_v.set_yticks([])

            # Row 1: Policy Arrows
            ax_p = axes[1, col]
            ax_p.imshow(base_img)
            for r in range(self.height):
                for c in range(self.width):
                    if self.walls[r, c]:
                        pass  # Solid clean wall tile without '#' character
                    elif (r, c) == box_pos:
                        ax_p.text(c, r, "B", ha="center", va="center", color="black", weight="bold", fontsize=10, bbox=dict(boxstyle="square,pad=0.2", facecolor="#FFA500", edgecolor="#333333", alpha=0.9))
                    elif (r, c) == self.goal_pos:
                        ax_p.text(c, r, "G", ha="center", va="center", color="white", weight="bold", fontsize=10, bbox=dict(boxstyle="square,pad=0.2", facecolor="#2E8B57", edgecolor="#333333", alpha=0.9))
                    elif hasattr(self.env, "hook_tile_pos") and (r, c) == self.env.hook_tile_pos:
                        ax_p.text(c, r, "H", ha="center", va="center", color="white", weight="bold", fontsize=10, bbox=dict(boxstyle="square,pad=0.2", facecolor="#7B1FA2", edgecolor="#333333", alpha=0.9))
                    else:
                        act = grid_pi[r, c]
                        if act >= 0:
                            arrow_char = self.ACTION_CHARS.get(act, "?")
                            ax_p.text(c, r, arrow_char, ha="center", va="center", color="#003366", fontsize=13, weight="bold")

            ax_p.set_title(r"Greedy Policy $\pi_{k}(s)$", fontsize=9.5, weight="normal")
            ax_p.grid(False)
            ax_p.set_xticks([])
            ax_p.set_yticks([])

        plt.suptitle(title, fontsize=10.5, weight="normal", y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=1.5)
        if os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_slice_comparison(
        self,
        V: np.ndarray,
        policy: np.ndarray,
        box_pos: Tuple[int, int],
        save_path: str,
        title: str = "Comparison of Value Function Slice: pull_available = 1 vs pull_available = 0",
    ):
        """
        Plots two slices side-by-side:
        Slice A: pull_available = 1 (can pull)
        Slice B: pull_available = 0 (cannot pull)
        Visually proves why pull completely changes state values around dead corners!
        """
        apply_ieee_style()
        grid_V_pull, grid_pi_pull = self.extract_slice_2d(V, policy, box_pos, pull_avail=1)
        grid_V_nopull, grid_pi_nopull = self.extract_slice_2d(V, policy, box_pos, pull_avail=0)

        fig, axes = plt.subplots(1, 2, figsize=(IEEE_DOUBLE_COL_WIDTH, 3.6))

        base_img = np.full((self.height, self.width, 3), 0.94)
        base_img[self.walls] = [0.22, 0.22, 0.25]

        # Slice 1: pull = 1
        axes[0].imshow(base_img)
        masked_pull = np.ma.masked_invalid(grid_V_pull)
        im0 = axes[0].imshow(masked_pull, cmap="Spectral", alpha=0.9)
        for r in range(self.height):
            for c in range(self.width):
                if self.walls[r, c]:
                    pass  # Solid clean wall tile without '#' character
                elif (r, c) == box_pos:
                    axes[0].text(c, r, "B", ha="center", va="center", color="black", weight="bold", fontsize=10, bbox=dict(boxstyle="square,pad=0.2", facecolor="#FFA500", edgecolor="#333333", alpha=0.9))
                elif (r, c) == self.goal_pos:
                    axes[0].text(c, r, "G", ha="center", va="center", color="white", weight="bold", fontsize=10, bbox=dict(boxstyle="square,pad=0.2", facecolor="#2E8B57", edgecolor="#333333", alpha=0.9))
                elif hasattr(self.env, "hook_tile_pos") and (r, c) == self.env.hook_tile_pos:
                    axes[0].text(c, r, "H", ha="center", va="center", color="white", weight="bold", fontsize=10, bbox=dict(boxstyle="square,pad=0.2", facecolor="#7B1FA2", edgecolor="#333333", alpha=0.9))
                elif not np.isnan(grid_V_pull[r, c]):
                    act = grid_pi_pull[r, c]
                    char = self.ACTION_CHARS.get(act, "")
                    axes[0].text(c, r, f"{grid_V_pull[r,c]:.1f}\n{char}", ha="center", va="center", color="black", fontsize=7.5, weight="bold")
        axes[0].set_title("Slice A: pull_available = 1\n(Pull Rescue Active - High Value near corners)", fontsize=9.5, weight="normal")
        axes[0].grid(False)
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        # Slice 2: pull = 0
        axes[1].imshow(base_img)
        masked_nopull = np.ma.masked_invalid(grid_V_nopull)
        im1 = axes[1].imshow(masked_nopull, cmap="Spectral", alpha=0.9)
        for r in range(self.height):
            for c in range(self.width):
                if self.walls[r, c]:
                    pass  # Solid clean wall tile without '#' character
                elif (r, c) == box_pos:
                    axes[1].text(c, r, "B", ha="center", va="center", color="black", weight="bold", fontsize=10, bbox=dict(boxstyle="square,pad=0.2", facecolor="#FFA500", edgecolor="#333333", alpha=0.9))
                elif (r, c) == self.goal_pos:
                    axes[1].text(c, r, "G", ha="center", va="center", color="white", weight="bold", fontsize=10, bbox=dict(boxstyle="square,pad=0.2", facecolor="#2E8B57", edgecolor="#333333", alpha=0.9))
                elif hasattr(self.env, "hook_tile_pos") and (r, c) == self.env.hook_tile_pos:
                    axes[1].text(c, r, "H", ha="center", va="center", color="white", weight="bold", fontsize=10, bbox=dict(boxstyle="square,pad=0.2", facecolor="#7B1FA2", edgecolor="#333333", alpha=0.9))
                elif not np.isnan(grid_V_nopull[r, c]):
                    act = grid_pi_nopull[r, c]
                    char = self.ACTION_CHARS.get(act, "")
                    axes[1].text(c, r, f"{grid_V_nopull[r,c]:.1f}\n{char}", ha="center", va="center", color="black", fontsize=7.5, weight="bold")
        axes[1].set_title("Slice B: pull_available = 0\n(Pull Exhausted - Red Dead Zones)", fontsize=9.5, weight="normal")
        axes[1].grid(False)
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        plt.suptitle(title, fontsize=10.0, weight="normal")
        plt.tight_layout()
        if os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
