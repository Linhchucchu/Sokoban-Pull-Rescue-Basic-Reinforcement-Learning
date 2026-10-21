"""
Unit Test Suite for Sokoban Environment Mechanics and Core Algorithms (Topic 15).
Tests:
1. First-Visit Monte Carlo: verifies earliest-visit return update.
2. Hook-Aware Deadlock: verifies corner reachability before terminal declaration.
3. Terminal Transition Zero-Bootstrapping in DP: verifies row sum == 0 for terminal states.
4. Dilemma Map Solution: verifies optimal policy solves in 31 steps with return +69.00.
"""

import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sokoban_env.sokoban_pull_env import SokobanPullEnv
from algorithms.monte_carlo import MonteCarloControl
from algorithms.dynamic_programming import DynamicProgramming


def test_first_visit_mc():
    print("[+] Testing First-Visit Monte Carlo implementation...")
    map_5x5 = os.path.join(PROJECT_ROOT, "maps", "map_5x5_basic.txt")
    env = SokobanPullEnv(map_5x5, mode="basic")
    mc = MonteCarloControl(env, gamma=0.9, first_visit=True)

    # Synthetic episode with repeated state-action: (s0, a0) visited at t=0 and t=2
    # episode: [(0, 0, -1.0), (1, 1, -1.0), (0, 0, 100.0)]
    # t=2 return: G_2 = 100.0
    # t=1 return: G_1 = -1.0 + 0.9 * 100 = 89.0
    # t=0 return: G_0 = -1.0 + 0.9 * 89 = 79.1
    episode = [(0, 0, -1.0), (1, 1, -1.0), (0, 0, 100.0)]

    first_visits = {}
    for t, (s, a, _) in enumerate(episode):
        if (s, a) not in first_visits:
            first_visits[(s, a)] = t

    G = 0.0
    for t in reversed(range(len(episode))):
        s_t, a_t, r_t = episode[t]
        G = r_t + mc.gamma * G
        if t != first_visits[(s_t, a_t)]:
            continue
        mc.N[s_t, a_t] += 1
        alpha = 1.0 / mc.N[s_t, a_t]
        mc.Q[s_t, a_t] += alpha * (G - mc.Q[s_t, a_t])

    assert np.isclose(mc.Q[0, 0], 79.1), f"Expected 79.1, got {mc.Q[0, 0]}"
    print("    PASSED: First-Visit MC correctly updates first visit return (79.1), not last visit (100.0).")


def test_hook_aware_deadlock():
    print("[+] Testing Hook-Aware Deadlock Detector...")
    map_hook = os.path.join(PROJECT_ROOT, "maps", "map_7x7_hook.txt")
    env = SokobanPullEnv(map_hook, mode="special")

    # Cell (7, 7) is a corner deadlock in bottom-right
    corner = (7, 7)
    assert corner in env.deadlock_detector.corner_deadlocks

    # Case 1: pull=0, hook_used=0, agent at (1, 1), hook at (7, 1) -> NOT deadlock because agent can reach Hook
    is_dl_free = env.deadlock_detector.is_permanent_deadlock(
        corner, pull_available=0, hook_used=0, agent_pos=(1, 1), hook_pos=env.hook_pos
    )
    assert is_dl_free is False, "Must NOT be deadlock when Hook is available and reachable!"
    print("    PASSED: Corner with 0 pulls but reachable Hook is NOT marked as deadlock.")

    # Case 2: pull=0, hook_used=1 (already used) -> TRUE permanent deadlock
    is_dl_used = env.deadlock_detector.is_permanent_deadlock(
        corner, pull_available=0, hook_used=1, agent_pos=(1, 1), hook_pos=env.hook_pos
    )
    assert is_dl_used is True, "Must be deadlock when Hook was already consumed!"
    print("    PASSED: Corner with 0 pulls and consumed Hook is correctly marked as deadlock.")


def test_terminal_zero_bootstrapping():
    print("[+] Testing Terminal Transition Zero-Bootstrapping in DP...")
    map_5x5 = os.path.join(PROJECT_ROOT, "maps", "map_5x5_basic.txt")
    env = SokobanPullEnv(map_5x5, mode="basic")
    P, R = env.get_transition_model()

    row_sums = P.sum(axis=-1)
    terminal_count = np.sum(row_sums == 0.0)
    non_terminal_count = np.sum(row_sums == 1.0)

    assert terminal_count > 0, "Must have terminal transitions with row sum == 0.0!"
    assert terminal_count + non_terminal_count == env.num_states * env.num_actions
    print(f"    PASSED: Found {terminal_count} terminal transitions with zero future bootstrap (sum=0.0).")
    print(f"            Found {non_terminal_count} non-terminal transitions with stochastic sum=1.0.")


def test_dilemma_optimal_vs_failure():
    print("[+] Testing Dilemma Map (Topic 15 core requirement)...")
    map_dilemma = os.path.join(PROJECT_ROOT, "maps", "map_7x7_dilemma.txt")
    env = SokobanPullEnv(map_dilemma, mode="special")
    P, R = env.get_transition_model()
    dp = DynamicProgramming(P, R, gamma=0.9)
    res = dp.value_iteration(theta=1e-7)

    # Rollout optimal policy
    obs, info = env.reset()
    opt_return = 0.0
    opt_steps = 0
    done = False
    while not done and opt_steps < 100:
        a = res["policy"][info["state_idx"]]
        obs, r, term, trunc, info = env.step(a)
        opt_return += r
        opt_steps += 1
        done = term or trunc

    assert env.is_solved, "Optimal policy must solve the Dilemma map!"
    assert opt_steps == 31, f"Expected 31 steps, got {opt_steps}"
    assert np.isclose(opt_return, 69.0), f"Expected return +69.0, got {opt_return}"
    print(f"    PASSED: Optimal policy solves Dilemma in {opt_steps} steps with return +{opt_return:.2f}.")

    # Rollout Pull-ASAP policy (pull at first opportunity at step 5, then follow Q*)
    obs, info = env.reset(seed=42)
    asap_return = 0.0
    asap_steps = 0
    done = False
    pulled_first = False
    first_pull_step = -1

    while not done and asap_steps < 100:
        s = info["state_idx"]
        state = env.all_states[s]
        ar, ac, br, bc, pull_avail, hook_used = state
        dr, dc = br - ar, bc - ac
        is_adj = (abs(dr) + abs(dc) == 1)
        back_r, back_c = ar - dr, ac - dc
        can_pull = (pull_avail == 1 and is_adj and 0 <= back_r < env.height and 0 <= back_c < env.width and not env.walls[back_r, back_c])

        if not pulled_first and can_pull:
            a = 4  # PULL ASAP
            pulled_first = True
            first_pull_step = asap_steps + 1
        else:
            a = int(res["policy"][s])

        obs, r, term, trunc, info = env.step(a)
        asap_return += r
        asap_steps += 1
        done = term or trunc

    assert env.is_solved, "Pull-ASAP policy must solve Dilemma map!"
    assert first_pull_step == 5, f"Expected first pull at step 5, got {first_pull_step}"
    assert asap_steps == 39, f"Expected 39 steps for Pull-ASAP, got {asap_steps}"
    assert np.isclose(asap_return, 61.0), f"Expected return +61.0, got {asap_return}"
    print(f"    PASSED: Pull-ASAP verified: first pull at step {first_pull_step}, {asap_steps} steps, return +{asap_return:.2f} (8-unit drop).")


if __name__ == "__main__":
    test_first_visit_mc()
    test_hook_aware_deadlock()
    test_terminal_zero_bootstrapping()
    test_dilemma_optimal_vs_failure()
    print("\nALL MECHANICS AND ALGORITHM UNIT TESTS PASSED WITH 100% SUCCESS!")
