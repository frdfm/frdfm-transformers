"""Conservative geometry masks for ten moves, computed from the input matrix.

Current body cells and already planned cells stay occupied for the entire plan.
This requires no body ordering, future observations, or random food positions.
The Transformer ranks actions that have a collision-free continuation.
"""
import numpy as np

from .game import DELTAS, TURNS


class PlanGeometry:
    def __init__(self, grid):
        heads = np.argwhere(grid >= 4)
        if len(heads) != 1:
            raise ValueError('Planning requires exactly one oriented head cell')
        self.position = tuple(map(int, heads[0]))
        self.direction = int(grid[self.position]) - 4
        self.free = set(map(tuple, np.argwhere((grid == 0) | (grid == 3)).tolist()))
        self.dead = False

    @staticmethod
    def destination(position, direction, action):
        direction = (direction + TURNS[action]) % 4
        dr, dc = DELTAS[direction]
        return (position[0] + dr, position[1] + dc), direction

    def can_finish(self, position, direction, free, steps):
        if steps == 0:
            return True
        if len(free) < steps:
            return False
        for action in range(3):
            p, heading = self.destination(position, direction, action)
            if p in free:
                free.remove(p)
                possible = self.can_finish(p, heading, free, steps - 1)
                free.add(p)
                if possible:
                    return True
        return False

    def allowed(self, remaining):
        """Prefer moves completing the whole horizon; otherwise allow safe steps.

        If completely trapped, return all actions so softmax remains defined.
        Geometry cannot rescue a snake with no legal move.
        """
        if self.dead:
            return [True] * 3
        immediate, complete = [], []
        for action in range(3):
            p, heading = self.destination(self.position, self.direction, action)
            safe = p in self.free
            immediate.append(safe)
            possible = False
            if safe:
                self.free.remove(p)
                possible = self.can_finish(p, heading, self.free, remaining - 1)
                self.free.add(p)
            complete.append(possible)
        return complete if any(complete) else immediate if any(immediate) else [True] * 3

    def advance(self, action):
        self.position, self.direction = self.destination(self.position, self.direction, action)
        if self.position in self.free:
            self.free.remove(self.position)
        else:
            self.dead = True
