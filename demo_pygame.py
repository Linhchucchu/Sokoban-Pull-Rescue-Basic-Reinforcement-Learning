"""
Interactive Pygame Demo for Sokoban Project 15 (Single Box with Pull Rescue & Hook).
Uses GameRenderer for High-Definition visual textures (3D bricks, cargo crate, glowing runes, worker).

Supports:
1. Autonomous Replay using Learned Optimal Policy (Q-table or VI Oracle)
2. Manual Play Mode (Keyboard Arrow Keys for Move/Push, SPACE or 'P' for Pull Rescue)
"""

import os
import sys
import argparse
import time
import pygame
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sokoban_env.sokoban_pull_env import SokobanPullEnv
from algorithms.dynamic_programming import DynamicProgramming
from visualization.game_renderer import GameRenderer


def find_or_solve_policy(env: SokobanPullEnv, map_name: str, policy_path: str = None) -> np.ndarray:
    """Loads existing policy file or solves via Value Iteration."""
    if policy_path and os.path.exists(policy_path):
        arr = np.load(policy_path)
        if arr.ndim == 2:
            return np.argmax(arr, axis=1)
        return arr

    base_map = os.path.splitext(os.path.basename(map_name))[0]
    candidates = [
        os.path.join(PROJECT_ROOT, "output", "tables", f"oracle_{env.mode}_{base_map}_policy.npy"),
        os.path.join(PROJECT_ROOT, "output", "tables", f"oracle_{env.mode}_{base_map}_Q.npy"),
        os.path.join(PROJECT_ROOT, "data", f"oracle_{env.mode}_{base_map}_policy.npy"),
        os.path.join(PROJECT_ROOT, "data", f"oracle_{env.mode}_{base_map}_Q.npy"),
    ]
    for c in candidates:
        if os.path.exists(c):
            print(f"[+] Loaded policy from: {c}")
            arr = np.load(c)
            return np.argmax(arr, axis=1) if arr.ndim == 2 else arr

    print(f"[*] Solving optimal policy via Value Iteration for {base_map}...")
    P, R = env.get_transition_model()
    dp = DynamicProgramming(P, R, gamma=0.9)
    vi = dp.value_iteration(theta=1e-7)
    return vi["policy"]


def run_demo(map_name="map_7x7_hook.txt", policy_path=None, manual=False, fps=4, cell_size=72):
    map_file = os.path.join(PROJECT_ROOT, "maps", map_name) if not os.path.isabs(map_name) else map_name
    base_name = os.path.splitext(os.path.basename(map_file))[0]
    mode = "basic" if "basic" in base_name.lower() else "special"

    env = SokobanPullEnv(map_file, mode=mode)
    renderer = GameRenderer(cell_size=cell_size)

    policy = None
    if not manual:
        try:
            policy = find_or_solve_policy(env, map_name=map_name, policy_path=policy_path)
            print("[+] Optimal Autonomous Policy loaded successfully.")
        except Exception as e:
            print(f"[!] Could not load/solve policy: {e}. Falling back to manual mode.")
            manual = True

    # Title mapping
    title_map = {
        "map_5x5_basic": "Setting 1: Basic 5x5 Sokoban",
        "map_5x5_inverted": "Variant 5x5: Inverted Open Map",
        "map_6x6_rescue": "Setting 3: Rescue Pull 6x6 Map",
        "map_6x6_labyrinth": "Variant 6x6: Labyrinth Maze Map",
        "map_7x7_deadcorners": "Setting 2: Dead Corners 7x7 Map",
        "map_7x7_hook": "Setting 4: Hook Recharge 7x7 Map",
        "map_7x7_dilemma": "Setting 5: Deadlock Dilemma 7x7 Map",
        "map_7x7_vault": "Variant 7x7: Central Fortified Vault",
        "map_7x7_zigzag": "Variant 7x7: Zigzag S-Curve Map",
    }
    map_title = title_map.get(base_name, f"Sokoban: {base_name}")

    # Initial frame to get window dimensions
    dummy_img = renderer.render_frame(env=env, map_title=map_title)
    win_w, win_h = dummy_img.size

    # Pygame Init
    pygame.init()
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption(f"Topic 15: Sokoban Pull Rescue [{map_title}]")
    clock = pygame.time.Clock()

    obs, info = env.reset()
    state = info["state_tuple"]
    total_reward = 0.0
    step_count = 0
    terminated = False
    running = True
    last_act = None
    last_event = "start"

    print("\n" + "=" * 60)
    print(f"Interactive Pygame Demo - {map_title}")
    print("=" * 60)
    if manual:
        print("  [Arrow Keys] : Move / Push cargo crate")
        print("  [P] / [SPACE]: Rescue Pull (Consumes pull credit)")
        print("  [R]          : Reset map")
        print("  [ESC]        : Exit")
    else:
        print("  Autonomous Agent executing optimal policy.")
        print("  [R]  : Restart episode")
        print("  [ESC]: Exit")
    print("=" * 60 + "\n")

    while running:
        action_to_take = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    state = info["state_tuple"]
                    total_reward = 0.0
                    step_count = 0
                    terminated = False
                    last_act = None
                    last_event = "reset"
                elif manual and not terminated:
                    if event.key == pygame.K_UP:
                        action_to_take = 0
                    elif event.key == pygame.K_DOWN:
                        action_to_take = 1
                    elif event.key == pygame.K_LEFT:
                        action_to_take = 2
                    elif event.key == pygame.K_RIGHT:
                        action_to_take = 3
                    elif event.key in [pygame.K_p, pygame.K_SPACE]:
                        action_to_take = 4

        # Autonomous Step
        if not manual and not terminated:
            s_idx = env.state_to_idx(state)
            action_to_take = int(policy[s_idx])
            time.sleep(1.0 / fps)

        # Execute Action
        if action_to_take is not None and not terminated:
            next_obs, r, term, trunc, step_info = env.step(action_to_take)
            total_reward += r
            step_count += 1
            state = step_info["state_tuple"]
            terminated = term or trunc
            last_act = action_to_take
            last_event = step_info.get("event", "")

        # Render Frame via GameRenderer
        pil_frame = renderer.render_frame(
            env=env,
            last_action=last_act,
            last_event=last_event,
            step_num=step_count,
            cum_reward=total_reward,
            is_solved=env.is_solved,
            is_terminated=terminated,
            map_title=map_title,
        )

        # Blit to Pygame window
        pg_surface = pygame.image.fromstring(pil_frame.tobytes(), pil_frame.size, pil_frame.mode)
        screen.blit(pg_surface, (0, 0))
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default="map_7x7_dilemma.txt")
    parser.add_argument("--policy", type=str, default=None)
    parser.add_argument("--manual", action="store_true")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--cell_size", type=int, default=72)
    args = parser.parse_args()

    run_demo(map_name=args.map, policy_path=args.policy, manual=args.manual, fps=args.fps, cell_size=args.cell_size)
