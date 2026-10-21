import os
import sys
from collections import deque
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sokoban_env.sokoban_pull_env import SokobanPullEnv

def count_all_reachable_states():
    print("Computing reachable state spaces across all settings via BFS...")
    settings = [
        {"setting": "Setting 1 (5x5 Basic)", "map": "map_5x5_basic.txt", "mode": "basic"},
        {"setting": "Intermediate (6x6 Rescue)", "map": "map_6x6_rescue.txt", "mode": "special"},
        {"setting": "Setting 2 (7x7 Deadcorners)", "map": "map_7x7_deadcorners.txt", "mode": "special"},
        {"setting": "Setting 3 (7x7 Dilemma No-Hook)", "map": "map_7x7_dilemma_nohook.txt", "mode": "special"},
        {"setting": "Setting 4 (7x7 Dilemma Hook)", "map": "map_7x7_dilemma.txt", "mode": "special"},
    ]

    records = []
    for cfg in settings:
        map_path = os.path.join(PROJECT_ROOT, "maps", cfg["map"])
        if not os.path.exists(map_path) and "nohook" in cfg["map"]:
            with open(os.path.join(PROJECT_ROOT, "maps", "map_7x7_dilemma.txt")) as f:
                c = f.read().replace("H", ".")
            with open(map_path, "w") as f:
                f.write(c)

        env = SokobanPullEnv(map_path, mode=cfg["mode"])
        P, R = env.get_transition_model()
        obs, info = env.reset(seed=42)
        s0 = info["state_idx"]

        visited = set([s0])
        queue = deque([s0])
        while queue:
            curr = queue.popleft()
            for a in range(env.num_actions):
                for nxt in np.where(P[curr, a] > 0)[0]:
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append(nxt)

        non_term_reachable = len(visited)
        represented_total = env.num_states
        unreachable_or_terminal = represented_total - non_term_reachable

        records.append({
            "setting": cfg["setting"],
            "map": cfg["map"],
            "mode": cfg["mode"],
            "actions": env.num_actions,
            "represented_states": represented_total,
            "non_terminal_reachable": non_term_reachable,
            "unreachable_or_terminal": unreachable_or_terminal,
            "reachable_pct": (non_term_reachable / represented_total) * 100.0
        })
        print(f"[{cfg['setting']:<30}] Represented={represented_total:5d} | Reachable={non_term_reachable:5d} ({non_term_reachable/represented_total*100:5.1f}%)")

    df = pd.DataFrame(records)
    out_csv = os.path.join(PROJECT_ROOT, "data", "reachable_states_summary.csv")
    df.to_csv(out_csv, index=False)
    out_table = os.path.join(PROJECT_ROOT, "output", "tables", "reachable_states_summary.csv")
    df.to_csv(out_table, index=False)
    print(f"Saved reachable state summary to {out_csv}")

if __name__ == "__main__":
    count_all_reachable_states()
