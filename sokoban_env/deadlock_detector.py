"""
Deadlock detector for Sokoban environments.
Detects corner deadlocks and line deadlocks.
"""

from typing import Set, Tuple, Optional
from collections import deque
import numpy as np


class DeadlockDetector:
    def __init__(self, walls: np.ndarray, goals: Set[Tuple[int, int]]):
        self.walls = walls
        self.goals = set(goals)
        self.height, self.width = walls.shape
        self.corner_deadlocks = self._find_corner_deadlocks()

    def _is_wall(self, r: int, c: int) -> bool:
        if not (0 <= r < self.height and 0 <= c < self.width):
            return True
        return bool(self.walls[r, c])

    def _find_corner_deadlocks(self) -> Set[Tuple[int, int]]:
        corners = set()
        for r in range(self.height):
            for c in range(self.width):
                if self.walls[r, c] or (r, c) in self.goals:
                    continue
                # Check for 4 possible corner wall configurations:
                up = self._is_wall(r - 1, c)
                down = self._is_wall(r + 1, c)
                left = self._is_wall(r, c - 1)
                right = self._is_wall(r, c + 1)

                if (up and left) or (up and right) or (down and left) or (down and right):
                    corners.add((r, c))
        return corners

    def can_agent_reach_hook(
        self,
        agent_pos: Tuple[int, int],
        hook_pos: Tuple[int, int],
        box_pos: Tuple[int, int],
    ) -> bool:
        """
        Check if the agent can reach the hook position without passing through
        walls or the box's current blocked position.
        """
        if agent_pos == hook_pos:
            return True
        queue = deque([agent_pos])
        visited = {agent_pos, box_pos}
        while queue:
            curr = queue.popleft()
            if curr == hook_pos:
                return True
            r, c = curr
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    if not self.walls[nr, nc] and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        return False

    def is_permanent_deadlock(
        self,
        box_pos: Tuple[int, int],
        pull_available: int = 0,
        hook_used: int = 1,
        agent_pos: Optional[Tuple[int, int]] = None,
        hook_pos: Optional[Tuple[int, int]] = None,
    ) -> bool:
        """
        Check if box is in an unrecoverable permanent deadlock.
        1. If box is in Goal -> NOT deadlock.
        2. If box is not in a corner deadlock -> NOT corner deadlock.
        3. If box is in a corner:
           - If pull_available > 0: NOT permanent deadlock (agent can pull box out).
           - If pull_available == 0:
             - If Hook is available (hook_pos is not None, hook_used == 0) and reachable:
               NOT permanent deadlock (agent can walk to Hook, recharge pull, and rescue box).
             - Otherwise: PERMANENT DEADLOCK.
        """
        if box_pos in self.goals:
            return False
        if box_pos not in self.corner_deadlocks:
            return False

        # Box is in a corner deadlock
        if pull_available > 0:
            return False

        # pull_available == 0: check if agent can recharge via Hook
        if hook_pos is not None and hook_used == 0:
            if agent_pos is not None:
                if self.can_agent_reach_hook(agent_pos, hook_pos, box_pos):
                    return False
            else:
                return False

        return True
