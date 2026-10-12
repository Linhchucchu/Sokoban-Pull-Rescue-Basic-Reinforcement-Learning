"""
Test basic environment and transition matrix.
"""

import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sokoban_env.sokoban_pull_env import SokobanPullEnv


def test_basic_env():
    map_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "maps", "map_5x5_basic.txt")
    env = SokobanPullEnv(map_path, mode="basic")
    
    print(f"Basic Env initialized!")
    print(f"Num Free Cells: {len(env.free_cells)}")
    print(f"Num States: {env.num_states}")
    print(f"Num Actions: {env.num_actions}")
    
    obs, info = env.reset()
    print("Initial render:")
    print(env.render("ansi"))
    
    # Test step
    next_obs, reward, term, trunc, info = env.step(3)  # RIGHT
    print(f"\nAfter RIGHT: reward={reward}, term={term}, event={info['event']}")
    print(env.render("ansi"))
    
    # Test transition model extraction
    P, R = env.get_transition_model()
    print(f"\nTransition Model extracted successfully!")
    print(f"P shape: {P.shape}, R shape: {R.shape}")
    # In an episodic MDP, terminal transitions have sum=0.0 (no future bootstrap)
    # and non-terminal transitions have sum=1.0.
    row_sums = P.sum(axis=-1)
    terminal_transitions = (row_sums == 0.0)
    non_terminal_transitions = (row_sums == 1.0)
    assert np.all(terminal_transitions | non_terminal_transitions), "All rows must sum to either 1.0 (non-terminal) or 0.0 (terminal)!"
    print(f"Non-terminal transitions: {np.sum(non_terminal_transitions)} (sum=1.0)")
    print(f"Terminal transitions: {np.sum(terminal_transitions)} (sum=0.0, correctly zero-bootstrapped)")
    print("ALL BASIC ENV TESTS PASSED!")


if __name__ == "__main__":
    test_basic_env()
