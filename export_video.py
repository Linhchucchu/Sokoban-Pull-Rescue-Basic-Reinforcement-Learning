"""
Export High-Definition MP4 and GIF Videos of Agent Gameplay for Topic 15.
Uses GameRenderer to produce beautiful HD frames and saves:
1. MP4 video (OpenCV)
2. Animated GIF (PIL)
"""

import os
import sys
import argparse
import shutil
import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sokoban_env.sokoban_pull_env import SokobanPullEnv
from algorithms.dynamic_programming import DynamicProgramming
from visualization.game_renderer import GameRenderer


def find_or_solve_policy(env: SokobanPullEnv, map_name: str, policy_path: str = None) -> np.ndarray:
    """Finds existing policy file or solves immediately using Value Iteration."""
    if policy_path and os.path.exists(policy_path):
        arr = np.load(policy_path)
        if arr.ndim == 2:
            return np.argmax(arr, axis=1)
        return arr

    base_map = os.path.splitext(os.path.basename(map_name))[0]
    # Search standard table locations
    candidates = [
        os.path.join(PROJECT_ROOT, "output", "tables", f"oracle_{env.mode}_{base_map}_policy.npy"),
        os.path.join(PROJECT_ROOT, "output", "tables", f"oracle_{env.mode}_{base_map}_Q.npy"),
        os.path.join(PROJECT_ROOT, "data", f"oracle_{env.mode}_{base_map}_policy.npy"),
        os.path.join(PROJECT_ROOT, "data", f"oracle_{env.mode}_{base_map}_Q.npy"),
    ]
    for c in candidates:
        if os.path.exists(c):
            print(f"[+] Found cached policy at: {c}")
            arr = np.load(c)
            return np.argmax(arr, axis=1) if arr.ndim == 2 else arr

    print(f"[*] Solving optimal policy via Value Iteration for {base_map} (|S|={env.num_states})...")
    P, R = env.get_transition_model()
    dp = DynamicProgramming(P, R, gamma=0.9)
    vi = dp.value_iteration(theta=1e-7)
    print(f"[+] Value Iteration converged in {vi['num_iterations']} iterations ({vi['elapsed_time']*1000:.1f} ms)!")
    os.makedirs(os.path.join(PROJECT_ROOT, "output", "tables"), exist_ok=True)
    np.save(os.path.join(PROJECT_ROOT, "output", "tables", f"oracle_{env.mode}_{base_map}_policy.npy"), vi["policy"])
    return vi["policy"]


def export_gameplay(
    map_name: str = "map_7x7_dilemma.txt",
    policy_path: str = None,
    output_dir: str = "output/videos",
    output_format: str = "both",
    fps: int = 4,
    cell_size: int = 72,
    hold_frames: int = 6,
    artifact_dir: str = None,
):
    map_file = os.path.join(PROJECT_ROOT, "maps", map_name) if not os.path.isabs(map_name) else map_name
    base_name = os.path.splitext(os.path.basename(map_file))[0]
    mode = "basic" if "basic" in base_name.lower() else "special"

    env = SokobanPullEnv(map_file, mode=mode)
    policy = find_or_solve_policy(env, map_name=map_name, policy_path=policy_path)

    renderer = GameRenderer(cell_size=cell_size)
    os.makedirs(os.path.join(PROJECT_ROOT, output_dir), exist_ok=True)

    # Pretty title for HUD
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

    print(f"\n[+] Recording rollout for '{map_title}'...")
    obs, info = env.reset()
    state = info["state_tuple"]
    frames = []

    # Initial frame
    f0 = renderer.render_frame(
        env=env,
        last_action=None,
        last_event="start",
        step_num=0,
        cum_reward=0.0,
        is_solved=False,
        is_terminated=False,
        map_title=map_title,
    )
    frames.append(f0)

    total_reward = 0.0
    step_num = 0
    max_steps = 100

    while step_num < max_steps:
        s_idx = env.state_to_idx(state)
        action = int(policy[s_idx])

        # Step environment
        next_obs, r, terminated, truncated, step_info = env.step(action)
        total_reward += r
        step_num += 1
        state = step_info["state_tuple"]
        event = step_info.get("event", "")

        is_solved = env.is_solved
        is_term = terminated or truncated

        frame = renderer.render_frame(
            env=env,
            last_action=action,
            last_event=event,
            step_num=step_num,
            cum_reward=total_reward,
            is_solved=is_solved,
            is_terminated=is_term,
            map_title=map_title,
        )
        frames.append(frame)

        if is_term:
            # Hold the final banner frame for a couple seconds
            for _ in range(hold_frames):
                frames.append(frame)
            break

    print(f"[+] Rollout completed: {step_num} steps | Return: {total_reward:+.1f} | Solved: {env.is_solved}")

    # Output paths
    mp4_path = os.path.join(PROJECT_ROOT, output_dir, f"{base_name}_gameplay.mp4")
    gif_path = os.path.join(PROJECT_ROOT, output_dir, f"{base_name}_gameplay.gif")

    w, h = frames[0].size

    # Save MP4
    if output_format in ["both", "mp4"]:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(mp4_path, fourcc, fps, (w, h))
        for img in frames:
            bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            writer.write(bgr)
        writer.release()
        print(f"[+] Saved MP4 video: {mp4_path} ({os.path.getsize(mp4_path) / 1024:.1f} KB)")

    # Save GIF
    if output_format in ["both", "gif"]:
        duration_ms = int(1000 / fps)
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
        )
        print(f"[+] Saved animated GIF: {gif_path} ({os.path.getsize(gif_path) / 1024:.1f} KB)")

        # Copy to artifact dir if provided
        if artifact_dir and os.path.exists(artifact_dir):
            dest = os.path.join(artifact_dir, f"{base_name}_gameplay.gif")
            shutil.copy2(gif_path, dest)
            print(f"[+] Copied GIF to artifact directory: {dest}")

    return mp4_path, gif_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export HD gameplay video and GIF for Sokoban Topic 15.")
    parser.add_argument("--map", type=str, default="map_7x7_dilemma.txt", help="Map file name in maps/")
    parser.add_argument("--policy", type=str, default=None, help="Optional policy .npy file")
    parser.add_argument("--format", type=str, default="both", choices=["both", "mp4", "gif"], help="Export format")
    parser.add_argument("--fps", type=int, default=4, help="Frames per second")
    parser.add_argument("--cell_size", type=int, default=72, help="Grid cell size in pixels")
    parser.add_argument("--hold_frames", type=int, default=6, help="Extra hold frames for final screen")
    parser.add_argument("--output_dir", type=str, default="output/videos", help="Directory to save videos")
    parser.add_argument("--artifact_dir", type=str, default=r"C:\Users\Admin\.gemini\antigravity\brain\dba0cb62-f2af-4952-a4c4-089a0280f93e", help="Artifact directory to copy GIF")
    args = parser.parse_args()

    export_gameplay(
        map_name=args.map,
        policy_path=args.policy,
        output_dir=args.output_dir,
        output_format=args.format,
        fps=args.fps,
        cell_size=args.cell_size,
        hold_frames=args.hold_frames,
        artifact_dir=args.artifact_dir,
    )
