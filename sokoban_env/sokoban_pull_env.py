"""
Custom Sokoban Gymnasium Environment with Pull Rescue and Hook Recharging.
Supports both Mode Basic (for Chapter 2, 3, 4 DP) and Mode Special (for Full De 15).
"""

from typing import Tuple, Set, Dict, Any, Optional, Union, List
import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .deadlock_detector import DeadlockDetector


class SokobanPullEnv(gym.Env):
    """
    Sokoban Environment for Project 15.
    
    Modes:
      - "basic": 4 actions (UP, DOWN, LEFT, RIGHT).
                 State: (r_A, c_A, r_B, c_B)
      - "special": 5 actions (UP, DOWN, LEFT, RIGHT, PULL).
                   State: (r_A, c_A, r_B, c_B, pull_available, hook_used)
    """

    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 4}

    ACTION_MOVES = {
        0: (-1, 0),  # UP
        1: (1, 0),   # DOWN
        2: (0, -1),  # LEFT
        3: (0, 1),   # RIGHT
    }
    ACTION_NAMES = {
        0: "UP",
        1: "DOWN",
        2: "LEFT",
        3: "RIGHT",
        4: "PULL",
    }

    def __init__(
        self,
        map_path_or_str: str,
        mode: str = "basic",  # "basic" or "special"
        max_steps: int = 100,
        step_penalty: float = -1.0,
        goal_reward: float = 100.0,
        deadlock_penalty: float = -20.0,
        invalid_pull_penalty: float = -1.0,
        pull_cost: float = -0.5,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.mode = mode
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.step_penalty = step_penalty
        self.goal_reward = goal_reward
        self.deadlock_penalty = deadlock_penalty
        self.invalid_pull_penalty = invalid_pull_penalty
        self.pull_cost = pull_cost

        # Load map
        self._load_map(map_path_or_str)

        # Action space
        if self.mode == "basic":
            self.action_space = spaces.Discrete(4)
        else:
            self.action_space = spaces.Discrete(5)

        # Deadlock detector
        self.deadlock_detector = DeadlockDetector(self.walls, {self.goal_pos})

        # Precompute state spaces & tabular indices
        self._build_state_spaces()

        # Observation space (Box of shape for Gym compatibility)
        if self.mode == "basic":
            self.observation_space = spaces.Box(
                low=np.array([0, 0, 0, 0]),
                high=np.array([self.height - 1, self.width - 1, self.height - 1, self.width - 1]),
                dtype=np.int32,
            )
        else:
            self.observation_space = spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0]),
                high=np.array([self.height - 1, self.width - 1, self.height - 1, self.width - 1, 1, 1]),
                dtype=np.int32,
            )

        # Runtime state
        self.agent_pos = self.initial_agent_pos
        self.box_pos = self.initial_box_pos
        self.pull_available = 1 if self.mode == "special" else 0
        self.hook_used = 0
        self.step_count = 0
        self.is_solved = False

    def _load_map(self, map_path_or_str: str):
        if os.path.isfile(map_path_or_str):
            with open(map_path_or_str, "r", encoding="utf-8") as f:
                lines = [line.rstrip("\r\n") for line in f if line.strip()]
        else:
            lines = [line.rstrip("\r\n") for line in map_path_or_str.strip().split("\n") if line.strip()]

        self.height = len(lines)
        self.width = max(len(line) for line in lines)
        self.walls = np.zeros((self.height, self.width), dtype=bool)
        self.free_cells: List[Tuple[int, int]] = []
        self.initial_agent_pos = None
        self.initial_box_pos = None
        self.goal_pos = None
        self.hook_pos = None

        for r, line in enumerate(lines):
            for c, char in enumerate(line):
                if char == "#":
                    self.walls[r, c] = True
                else:
                    self.free_cells.append((r, c))
                    if char == "S":
                        self.initial_agent_pos = (r, c)
                    elif char == "B":
                        self.initial_box_pos = (r, c)
                    elif char == "G":
                        self.goal_pos = (r, c)
                    elif char == "H":
                        self.hook_pos = (r, c)

        if self.initial_agent_pos is None or self.initial_box_pos is None or self.goal_pos is None:
            raise ValueError("Map must contain S (Agent), B (Box), and G (Goal).")

    def _build_state_spaces(self):
        self.all_states = []
        self.state_to_idx_dict: Dict[Tuple, int] = {}
        self.idx_to_state_dict: Dict[int, Tuple] = {}

        idx = 0
        for ar, ac in self.free_cells:
            for br, bc in self.free_cells:
                if (ar, ac) == (br, bc):
                    continue
                if self.mode == "basic":
                    st = (ar, ac, br, bc)
                    self.all_states.append(st)
                    self.state_to_idx_dict[st] = idx
                    self.idx_to_state_dict[idx] = st
                    idx += 1
                else:
                    for pull in [0, 1]:
                        for hook in [0, 1]:
                            st = (ar, ac, br, bc, pull, hook)
                            self.all_states.append(st)
                            self.state_to_idx_dict[st] = idx
                            self.idx_to_state_dict[idx] = st
                            idx += 1

        self.num_states = len(self.all_states)
        self.num_actions = self.action_space.n

    def state_to_idx(self, state: Tuple) -> int:
        return self.state_to_idx_dict[state]

    def idx_to_state(self, idx: int) -> Tuple:
        return self.idx_to_state_dict[idx]

    def get_all_states(self) -> List[Tuple]:
        return self.all_states

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.agent_pos = self.initial_agent_pos
        self.box_pos = self.initial_box_pos
        self.pull_available = 1 if self.mode == "special" else 0
        self.hook_used = 0
        self.step_count = 0
        self.is_solved = False

        state = self._get_state_tuple()
        info = {
            "state_tuple": state,
            "state_idx": self.state_to_idx(state),
            "is_solved": False,
            "step_count": 0,
        }
        return np.array(state, dtype=np.int32), info

    def _get_state_tuple(self) -> Tuple:
        if self.mode == "basic":
            return (self.agent_pos[0], self.agent_pos[1], self.box_pos[0], self.box_pos[1])
        else:
            return (
                self.agent_pos[0],
                self.agent_pos[1],
                self.box_pos[0],
                self.box_pos[1],
                self.pull_available,
                self.hook_used,
            )

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        current_state = self._get_state_tuple()
        next_state, reward, terminated, truncated, info = self.simulate_step(current_state, action)
        
        self.step_count += 1
        if self.step_count >= self.max_steps and not terminated:
            truncated = True

        # Apply state
        self.agent_pos = (next_state[0], next_state[1])
        self.box_pos = (next_state[2], next_state[3])
        if self.mode == "special":
            self.pull_available = next_state[4]
            self.hook_used = next_state[5]

        self.is_solved = (self.box_pos == self.goal_pos)
        info["step_count"] = self.step_count
        info["state_idx"] = self.state_to_idx(next_state)

        return np.array(next_state, dtype=np.int32), reward, terminated, truncated, info

    def simulate_step(self, state: Tuple, action: int) -> Tuple[Tuple, float, bool, bool, Dict[str, Any]]:
        """
        Deterministic transition function: (s, a) -> (s', r, terminated, truncated, info)
        Used by step() and Dynamic Programming (Bellman backups).
        """
        if self.mode == "basic":
            ar, ac, br, bc = state
            pull_avail, hook_used = 0, 0
        else:
            ar, ac, br, bc, pull_avail, hook_used = state

        # If already at goal or permanent deadlock, terminal absorbing state
        if (br, bc) == self.goal_pos:
            return state, 0.0, True, False, {"is_solved": True, "event": "terminal_goal"}
        if self.deadlock_detector.is_permanent_deadlock(
            (br, bc),
            pull_avail,
            hook_used=hook_used,
            agent_pos=(ar, ac),
            hook_pos=self.hook_pos,
        ):
            return state, 0.0, True, False, {"is_solved": False, "event": "terminal_deadlock"}

        agent_pos = (ar, ac)
        box_pos = (br, bc)
        reward = self.step_penalty
        terminated = False
        truncated = False
        event = "move"

        # Action 0..3: Standard Move/Push
        if action in [0, 1, 2, 3]:
            dr, dc = self.ACTION_MOVES[action]
            next_agent = (ar + dr, ac + dc)

            if not (0 <= next_agent[0] < self.height and 0 <= next_agent[1] < self.width) or self.walls[next_agent]:
                # Va vào tường
                pass
            elif next_agent == box_pos:
                # Đẩy thùng
                next_box = (br + dr, bc + dc)
                if (0 <= next_box[0] < self.height and 0 <= next_box[1] < self.width) and not self.walls[next_box]:
                    agent_pos = next_agent
                    box_pos = next_box
                    event = "push"
                else:
                    # Thùng bị cản bởi tường
                    pass
            else:
                agent_pos = next_agent
                event = "move"

        # Action 4: Pull Rescue (Only in mode="special")
        elif action == 4 and self.mode == "special":
            if pull_avail > 0:
                # Tìm hướng của box so với agent
                dr = box_pos[0] - agent_pos[0]
                dc = box_pos[1] - agent_pos[1]
                is_adjacent = (abs(dr) + abs(dc) == 1)

                if is_adjacent:
                    # Hướng lùi của agent ngược với hướng của box
                    back_agent = (agent_pos[0] - dr, agent_pos[1] - dc)
                    if (0 <= back_agent[0] < self.height and 0 <= back_agent[1] < self.width) and not self.walls[back_agent]:
                        # Kéo thành công!
                        new_box = agent_pos
                        new_agent = back_agent
                        agent_pos = new_agent
                        box_pos = new_box
                        pull_avail = 0
                        reward = self.step_penalty + self.pull_cost
                        event = "pull_success"
                    else:
                        reward = self.step_penalty + self.invalid_pull_penalty
                        event = "pull_blocked"
                else:
                    reward = self.step_penalty + self.invalid_pull_penalty
                    event = "pull_not_adjacent"
            else:
                reward = self.step_penalty + self.invalid_pull_penalty
                event = "pull_no_credit"

        # Check Hook recharging (Setting 4)
        if self.mode == "special" and self.hook_pos is not None:
            if agent_pos == self.hook_pos and pull_avail == 0 and hook_used == 0:
                pull_avail = 1
                hook_used = 1
                event += "_hook_recharged"

        # Check Goal condition
        if box_pos == self.goal_pos:
            reward = self.goal_reward
            terminated = True
            event = "goal_reached"

        # Check Deadlock condition
        elif self.deadlock_detector.is_permanent_deadlock(
            box_pos,
            pull_avail,
            hook_used=hook_used,
            agent_pos=agent_pos,
            hook_pos=self.hook_pos,
        ):
            reward = self.deadlock_penalty
            terminated = True
            event = "deadlock"

        # Pack next state
        if self.mode == "basic":
            next_state = (agent_pos[0], agent_pos[1], box_pos[0], box_pos[1])
        else:
            next_state = (agent_pos[0], agent_pos[1], box_pos[0], box_pos[1], pull_avail, hook_used)

        info = {
            "state_tuple": next_state,
            "event": event,
            "is_solved": (box_pos == self.goal_pos),
        }
        return next_state, reward, terminated, truncated, info

    def get_transition_model(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts full Transition Matrix P[s, a, s'] and Reward Matrix R[s, a]
        for Model-Based Dynamic Programming (Chapters 2, 3, 4).
        Terminal transitions zero out future states (P[s, a, :] = 0) to prevent
        erroneous bootstrapping past episode termination (Goal or Deadlock).
        """
        num_s = self.num_states
        num_a = self.num_actions

        P = np.zeros((num_s, num_a, num_s), dtype=np.float64)
        R = np.zeros((num_s, num_a), dtype=np.float64)

        for s_idx, state in enumerate(self.all_states):
            for a in range(num_a):
                next_st, r, terminated, _, _ = self.simulate_step(state, a)
                R[s_idx, a] = r
                if not terminated:
                    next_idx = self.state_to_idx(next_st)
                    P[s_idx, a, next_idx] = 1.0
                # If terminated, P[s_idx, a, :] remains 0.0 (no future bootstrapping)

        return P, R

    def render(self, mode: str = "ansi") -> Optional[np.ndarray]:
        grid = []
        for r in range(self.height):
            row = []
            for c in range(self.width):
                if self.walls[r, c]:
                    row.append("#")
                elif (r, c) == self.agent_pos and (r, c) == self.goal_pos:
                    row.append("+")
                elif (r, c) == self.box_pos and (r, c) == self.goal_pos:
                    row.append("*")
                elif (r, c) == self.agent_pos:
                    row.append("S")
                elif (r, c) == self.box_pos:
                    row.append("B")
                elif (r, c) == self.goal_pos:
                    row.append("G")
                elif self.hook_pos is not None and (r, c) == self.hook_pos:
                    row.append("H")
                else:
                    row.append(".")
            grid.append("".join(row))

        if mode == "ansi":
            return "\n".join(grid)
        elif mode == "rgb_array":
            # Simple RGB rendering for Colab / Video
            cell_size = 40
            img = np.zeros((self.height * cell_size, self.width * cell_size, 3), dtype=np.uint8)
            colors = {
                "#": (50, 50, 50),     # Wall: Dark Gray
                ".": (230, 230, 230), # Floor: Light Gray
                "S": (0, 150, 255),   # Agent: Blue
                "B": (220, 120, 0),   # Box: Orange
                "G": (0, 200, 50),    # Goal: Green
                "*": (255, 215, 0),   # Box on Goal: Gold
                "H": (200, 0, 200),   # Hook: Purple
            }
            for r in range(self.height):
                for c in range(self.width):
                    char = grid[r][c]
                    color = colors.get(char, (200, 200, 200))
                    img[r*cell_size:(r+1)*cell_size, c*cell_size:(c+1)*cell_size] = color
            return img
        return None
