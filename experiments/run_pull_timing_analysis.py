import os
import sys
from collections import deque
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sokoban_env.sokoban_pull_env import SokobanPullEnv
from algorithms.dynamic_programming import DynamicProgramming

def run_pull_timing_analysis():
    print("Executing Pull Timing & Value Deterioration Analysis on Dilemma Map...")
    env = SokobanPullEnv(os.path.join(PROJECT_ROOT, "maps", "map_7x7_dilemma.txt"), mode="special")
    P, R = env.get_transition_model()
    dp = DynamicProgramming(P, R, gamma=0.9)
    res = dp.value_iteration(theta=1e-7)
    Q = res["Q"]
    V = res["V"]
    
    # 1. BFS to find reachable state subset
    obs, info = env.reset(seed=42)
    start_idx = info["state_idx"]
    visited = set([start_idx])
    queue = deque([start_idx])
    while queue:
        curr = queue.popleft()
        for a in range(env.num_actions):
            next_indices = np.where(P[curr, a] > 0)[0]
            for nxt in next_indices:
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
                    
    # 2. Analyze states where PULL is authorized and physically valid
    valid_pull_states = []
    subopt_5_count = 0
    records = []
    
    for s_idx in visited:
        state = env.all_states[s_idx]
        ar, ac, br, bc, pull_avail, hook_used = state
        if pull_avail == 1:
            dr, dc = br - ar, bc - ac
            if abs(dr) + abs(dc) == 1:
                back_r, back_c = ar - dr, ac - dc
                if 0 <= back_r < env.height and 0 <= back_c < env.width and not env.walls[back_r, back_c]:
                    valid_pull_states.append(s_idx)
                    q_pull = float(Q[s_idx, 4])
                    q_max = float(np.max(Q[s_idx]))
                    diff = q_max - q_pull
                    is_subopt_5 = diff > 5.0
                    if is_subopt_5:
                        subopt_5_count += 1
                    records.append({
                        "state_idx": s_idx,
                        "agent_r": ar, "agent_c": ac,
                        "box_r": br, "box_c": bc,
                        "hook_used": hook_used,
                        "q_pull": q_pull,
                        "q_optimal": q_max,
                        "value_drop": diff,
                        "drop_exceeds_5": is_subopt_5
                    })
                    
    df = pd.DataFrame(records)
    out_csv = os.path.join(PROJECT_ROOT, "data", "pull_timing_analysis.csv")
    df.to_csv(out_csv, index=False)
    out_table = os.path.join(PROJECT_ROOT, "output", "tables", "pull_timing_analysis.csv")
    df.to_csv(out_table, index=False)
    
    print(f"Total reachable states: {len(visited)} / {env.num_states}")
    print(f"Reachable valid PULL states: {len(valid_pull_states)}")
    print(f"States where Q*(s, PULL) < max Q* - 5.0: {subopt_5_count} / {len(valid_pull_states)} ({subopt_5_count/len(valid_pull_states)*100:.1f}%)")
    print(f"Saved analysis to {out_csv}")

if __name__ == "__main__":
    run_pull_timing_analysis()
