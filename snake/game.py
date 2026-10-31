"""Seeded grid Snake and a recording-only shortest-path teacher."""
from collections import deque
from dataclasses import dataclass
import random

import numpy as np

STRAIGHT, LEFT, RIGHT = range(3)
ACTIONS = ('FORWARD', 'LEFT', 'RIGHT')
TURNS = (0, -1, 1)
DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))


@dataclass(frozen=True)
class Config:
    height: int = 20
    width: int = 20
    history: int = 2

    def __post_init__(self):
        if min(self.height, self.width) < 8 or self.history < 1:
            raise ValueError('Board must be at least 8x8; history must be positive')


class Snake:
    def __init__(self, seed=0, config=None):
        self.config = config or Config()
        self.rng = random.Random(seed)
        c = self.config
        self.direction = self.rng.randrange(4)
        dr, dc = DELTAS[self.direction]
        # All four initial headings fit without touching the wall.
        head = (self.rng.randrange(3, c.height - 3), self.rng.randrange(3, c.width - 3))
        self.body = [head, (head[0] - dr, head[1] - dc), (head[0] - 2*dr, head[1] - 2*dc)]
        self.score = self.ticks = 0
        self.alive, self.won = True, False
        self.food = self.spawn_food()
        self.frames = deque((self.matrix() for _ in range(c.history)), maxlen=c.history)

    def spawn_food(self):
        c = self.config
        occupied = set(self.body)
        empty = [(r, col) for r in range(1, c.height - 1)
                 for col in range(1, c.width - 1) if (r, col) not in occupied]
        return self.rng.choice(empty) if empty else None

    def matrix(self):
        """0 empty, 1 wall, 2 body, 3 food, 4..7 head facing U/R/D/L."""
        c = self.config
        grid = np.zeros((c.height, c.width), dtype=np.uint8)
        grid[[0, -1], :] = 1
        grid[:, [0, -1]] = 1
        for p in self.body:
            grid[p] = 2
        if self.food is not None:
            grid[self.food] = 3
        grid[self.body[0]] = 4 + self.direction
        return grid

    def observation(self):
        return np.stack(self.frames)

    def destination(self, action):
        if action not in (STRAIGHT, LEFT, RIGHT):
            raise ValueError(f'Invalid action: {action}')
        direction = (self.direction + TURNS[action]) % 4
        dr, dc = DELTAS[direction]
        return (self.body[0][0] + dr, self.body[0][1] + dc), direction

    def safe(self, action):
        p, _ = self.destination(action)
        grows = p == self.food
        return (0 < p[0] < self.config.height - 1 and 0 < p[1] < self.config.width - 1
                and p not in (self.body if grows else self.body[:-1]))

    def step(self, action):
        """One cell per tick; never auto-reset or override predicted actions."""
        if not self.alive or self.won:
            return False
        p, direction = self.destination(action)
        self.ticks += 1
        if not self.safe(action):
            self.alive = False
            return False
        self.direction = direction
        self.body.insert(0, p)
        ate = p == self.food
        if ate:
            self.score += 1
            self.food = self.spawn_food()
            self.won = self.food is None
        else:
            self.body.pop()
        self.frames.append(self.matrix())
        return ate


class Session:
    """Fresh random games on restart; an explicit seed reproduces the session."""
    def __init__(self, config=None, seed=None):
        self.config = config or Config()
        self.rng = random.Random(seed)
        self.seed = seed
        self.restart(first=True)

    def restart(self, first=False):
        if not first or self.seed is None:
            previous = self.seed
            self.seed = self.rng.getrandbits(63)
            while self.seed == previous:
                self.seed = self.rng.getrandbits(63)
        self.game = Snake(self.seed, self.config)
        return self.game


def expert(game):
    """BFS to visible food; otherwise choose the largest reachable free area.

    This teacher is used only to record labels or as an explicit benchmark.
    It does not know future food positions and is not a perfect Snake solver.
    """
    head = game.body[0]
    blocked = set(game.body[1:-1])
    queue = deque()
    visited = {head}
    for action in (STRAIGHT, LEFT, RIGHT):
        if game.safe(action):
            p, _ = game.destination(action)
            queue.append((p, action))
            visited.add(p)
    while queue and game.food is not None:
        p, first = queue.popleft()
        if p == game.food:
            return first
        for dr, dc in DELTAS:
            q = p[0] + dr, p[1] + dc
            if (0 < q[0] < game.config.height - 1 and 0 < q[1] < game.config.width - 1
                    and q not in blocked and q not in visited):
                visited.add(q)
                queue.append((q, first))
    def area(action):
        if not game.safe(action):
            return -1
        start, _ = game.destination(action)
        seen, todo = {start}, [start]
        for r, col in todo:
            for dr, dc in DELTAS:
                p = r + dr, col + dc
                if (0 < p[0] < game.config.height - 1 and 0 < p[1] < game.config.width - 1
                        and p not in blocked and p not in seen and p != head):
                    seen.add(p)
                    todo.append(p)
        return len(seen)
    return max((STRAIGHT, LEFT, RIGHT), key=area)
