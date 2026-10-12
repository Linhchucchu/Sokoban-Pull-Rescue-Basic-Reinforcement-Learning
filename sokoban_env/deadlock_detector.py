"""
Deadlock detector for Sokoban environments.
Detects corner deadlocks and line deadlocks.
"""

from typing import Set, Tuple
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

    def is_permanent_deadlock(self, box_pos: Tuple[int, int], pull_available: int = 0) -> bool:
        """
        Check if box is in a deadlock.
        If pull_available > 0, corner is NOT permanent deadlock because agent can pull box out.
        If pull_available == 0, corner is permanent deadlock.
        """
        if box_pos in self.goals:
            return False
        if pull_available > 0:
            return False
        return box_pos in self.corner_deadlocks
