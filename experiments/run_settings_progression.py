import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sokoban_env.sokoban_pull_env import SokobanPullEnv
from algorithms.temporal_difference import TemporalDifferenceControl

def run_settings_progression():
    print("Running Settings Progression Benchmarks across fine-grained checkpoints...")
    
    settings = [
        {"name": "Setting 1 (5x5 Basic)", "map": "map_5x5_basic.txt", "mode": "basic", "checkpoints": [250, 500, 1000]},
        {"name": "Intermediate (6x6 Rescue)", "map": "map_6x6_rescue.txt", "mode": "special", "checkpoints": [500, 1000, 2000, 3000, 5000]},
        {"name": "Setting 2 (7x7 Deadcorners)", "map": "map_7x7_deadcorners.txt", "mode": "special", "checkpoints": [1000, 2000, 3000, 5000, 10000]},
        {"name": "Setting 3 (7x7 Dilemma No-Hook)", "map": "map_7x7_dilemma_nohook.txt", "mode": "special", "checkpoints": [5000, 10000, 15000, 30000]},
        {"name": "Setting 4 (7x7 Dilemma Hook)", "map": "map_7x7_dilemma.txt", "mode": "special", "checkpoints": [5000, 10000, 30000]}
    ]
    
    seeds = [42, 100, 2024]
    records = []
    
    for cfg in settings:
        map_path = os.path.join(PROJECT_ROOT, "maps", cfg["map"])
        if not os.path.exists(map_path) and "nohook" in cfg["map"]:
            # Auto-generate nohook map from dilemma
            with open(os.path.join(PROJECT_ROOT, "maps", "map_7x7_dilemma.txt")) as f:
                c = f.read().replace("H", ".")
            with open(map_path, "w") as f:
                f.write(c)
                
        env = SokobanPullEnv(map_path, mode=cfg["mode"])
        checkpoints = sorted(cfg["checkpoints"])
        max_cp = max(checkpoints)
        
        for seed in seeds:
            np.random.seed(seed)
            agent = TemporalDifferenceControl(
                env,
                algo="q_learning",
                gamma=0.9,
                alpha=0.1,
                epsilon_start=1.0,
                epsilon_min=0.05,
                epsilon_decay=0.9995
            )
            
            # Incremental checkpoint evaluation
            current_eps = 0
            for cp in checkpoints:
                train_needed = cp - current_eps
                if train_needed > 0:
                    agent.train(num_episodes=train_needed)
                    current_eps = cp
                
                # Evaluate greedy
                obs, info = env.reset(seed=42)
                eval_steps = 0
                eval_ret = 0.0
                eval_pulls = 0
                done = False
                while not done and eval_steps < 100:
                    q_s = agent.Q[info["state_idx"]]
                    if np.all(q_s == q_s[0]):
                        a = np.random.randint(env.num_actions)
                    else:
                        a = int(np.argmax(q_s))
                    if a == 4:
                        eval_pulls += 1
                    obs, r, term, trunc, info = env.step(a)
                    eval_steps += 1
                    eval_ret += r
                    done = term or trunc
                    
                records.append({
                    "setting": cfg["name"],
                    "map": cfg["map"],
                    "mode": cfg["mode"],
                    "seed": seed,
                    "episodes": cp,
                    "greedy_solved": env.is_solved,
                    "greedy_steps": eval_steps,
                    "greedy_return": eval_ret,
                    "greedy_pulls": eval_pulls
                })
                print(f"[{cfg['name']}] Seed {seed} @ {cp:5d} eps -> Solved={env.is_solved}, Steps={eval_steps}, Ret={eval_ret:.1f}")

    df = pd.DataFrame(records)
    out_csv = os.path.join(PROJECT_ROOT, "data", "settings_convergence_progression.csv")
    df.to_csv(out_csv, index=False)
    out_table = os.path.join(PROJECT_ROOT, "output", "tables", "settings_convergence_progression.csv")
    df.to_csv(out_table, index=False)
    print(f"Saved progression benchmark to {out_csv}")

if __name__ == "__main__":
    run_settings_progression()
