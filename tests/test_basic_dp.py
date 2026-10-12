"""
Test basic environment and transition matrix.
"""

import os
import sys
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
    print(f"Sum of probabilities for all (s, a): {P.sum(axis=-1).min()} to {P.sum(axis=-1).max()}")
    assert P.sum(axis=-1).min() == 1.0, "All transition probabilities must sum to 1.0!"
    assert P.sum(axis=-1).max() == 1.0, "All transition probabilities must sum to 1.0!"
    print("ALL BASIC ENV TESTS PASSED!")


if __name__ == "__main__":
    test_basic_env()
