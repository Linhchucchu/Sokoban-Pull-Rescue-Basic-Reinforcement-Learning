import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sokoban_env.sokoban_pull_env import SokobanPullEnv

def run_random_baseline(num_episodes=20000):
    print(f"Executing Uniform Random Baseline for {num_episodes} episodes on Dilemma map...")
    env = SokobanPullEnv(os.path.join(PROJECT_ROOT, "maps", "map_7x7_dilemma.txt"), mode="special")
    
    solved_count = 0
    timeout_count = 0
    deadlock_count = 0
    returns = []
    steps_list = []
    
    for ep in range(num_episodes):
        obs, info = env.reset(seed=ep)
        ep_ret = 0.0
        step_count = 0
        for st in range(100):
            a = np.random.randint(0, 5)
            obs, r, term, trunc, info = env.step(a)
            ep_ret += r
            step_count += 1
            if term or trunc:
                if env.is_solved:
                    solved_count += 1
                elif trunc:
                    timeout_count += 1
                else:
                    deadlock_count += 1
                break
        returns.append(ep_ret)
        steps_list.append(step_count)
    
    df = pd.DataFrame([{
        "total_episodes": num_episodes,
        "solved_count": solved_count,
        "solved_pct": (solved_count / num_episodes) * 100.0,
        "timeout_count": timeout_count,
        "timeout_pct": (timeout_count / num_episodes) * 100.0,
        "deadlock_count": deadlock_count,
        "deadlock_pct": (deadlock_count / num_episodes) * 100.0,
        "mean_return": np.mean(returns),
        "std_return": np.std(returns),
        "mean_steps": np.mean(steps_list),
        "std_steps": np.std(steps_list)
    }])
    
    out_csv = os.path.join(PROJECT_ROOT, "data", "random_policy_baseline.csv")
    df.to_csv(out_csv, index=False)
    out_table = os.path.join(PROJECT_ROOT, "output", "tables", "random_policy_baseline.csv")
    df.to_csv(out_table, index=False)
    print(f"Saved random baseline to {out_csv}")
    print(df.to_string())

if __name__ == "__main__":
    run_random_baseline()
