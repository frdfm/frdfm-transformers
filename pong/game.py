"""Deterministic grid Pong. No renderer or ML dependencies beyond NumPy."""
from collections import deque
from dataclasses import dataclass
import random

import numpy as np

STAY, UP, DOWN = range(3)
ACTIONS = ('STAY', 'UP', 'DOWN')


@dataclass(frozen=True)
class Config:
    height: int = 24
    width: int = 32
    paddle_height: int = 5
    history: int = 4
    paddle_speed: float = 0.8
    ball_speed: float = 0.7

    def __post_init__(self):
        if self.height < 12 or self.width < 16:
            raise ValueError('Court must be at least 12 by 16')
        if not 1 <= self.paddle_height < self.height - 2 or self.history < 2:
            raise ValueError('Invalid paddle height or history (minimum 2)')
        if not 0 < self.ball_speed <= 1 or not 0 < self.paddle_speed <= 1:
            raise ValueError('Speeds must be in (0, 1] to prevent tunneling')


class Pong:
    def __init__(self, seed=0, config=None):
        self.config = config or Config()
        self.rng = random.Random(seed)
        self.score = [0, 0]
        self.hits = [0, 0]
        self.frames = deque(maxlen=self.config.history)
        self.reset_rally()

    def reset_rally(self):
        c = self.config
        self.paddles = [(c.height - 1) / 2] * 2
        self.x, self.y = (c.width - 1) / 2, self.rng.uniform(3, c.height - 4)
        self.vx = self.rng.choice((-1, 1)) * c.ball_speed
        self.vy = self.rng.choice((-1, 1)) * self.rng.uniform(0.3, 0.8)
        self.frames.clear()
        for _ in range(c.history):
            self.frames.append(self.matrix())

    def matrix(self):
        """0 empty, 1 wall, 2 paddle, 3 ball. Square ball occupies one cell."""
        c = self.config
        grid = np.zeros((c.height, c.width), dtype=np.uint8)
        grid[[0, -1], :] = 1
        for side, x in enumerate((1, c.width - 2)):
            top = round(self.paddles[side] - (c.paddle_height - 1) / 2)
            grid[top:top + c.paddle_height, x] = 2
        grid[int(np.clip(round(self.y), 1, c.height - 2)),
             int(np.clip(round(self.x), 0, c.width - 1))] = 3
        return grid

    def observation(self, side=0):
        """Both players see themselves on the left; time is oldest to newest."""
        if side not in (0, 1):
            raise ValueError('side must be 0 or 1')
        frames = np.stack(self.frames)
        return frames if side == 0 else frames[:, :, ::-1].copy()

    def step(self, left, right):
        """Advance one tick. Return scoring side or None; reset after a point."""
        c = self.config
        low = 1 + (c.paddle_height - 1) / 2
        high = c.height - 2 - (c.paddle_height - 1) / 2
        for side, action in enumerate((left, right)):
            if action not in (STAY, UP, DOWN):
                raise ValueError(f'Invalid action: {action}')
            delta = (0, -c.paddle_speed, c.paddle_speed)[action]
            self.paddles[side] = float(np.clip(self.paddles[side] + delta, low, high))
        old_x = self.x
        self.x += self.vx
        self.y += self.vy
        if self.y < 1:
            self.y, self.vy = 2 - self.y, abs(self.vy)
        elif self.y > c.height - 2:
            self.y, self.vy = 2 * (c.height - 2) - self.y, -abs(self.vy)
        for side, plane in enumerate((2, c.width - 3)):
            crossed = old_x > plane >= self.x if side == 0 else old_x < plane <= self.x
            if crossed and abs(self.y - self.paddles[side]) <= c.paddle_height / 2:
                self.x = 2 * plane - self.x
                self.vx = abs(self.vx) * (1 if side == 0 else -1)
                offset = (self.y - self.paddles[side]) / (c.paddle_height / 2)
                self.vy = float(np.clip(self.vy + offset * 0.3, -0.9, 0.9))
                self.hits[side] += 1
        scorer = 1 if self.x < 0 else 0 if self.x > c.width - 1 else None
        if scorer is not None:
            self.score[scorer] += 1
            self.reset_rally()
        else:
            self.frames.append(self.matrix())
        return scorer


def expert(game, side=0):
    """Privileged teacher: reflect a predicted intercept off the court walls.

    Only labels use physics state. Learned policies receive matrices exclusively.
    """
    c = game.config
    incoming = game.vx < 0 if side == 0 else game.vx > 0
    target = (c.height - 1) / 2
    if incoming:
        plane = 2 if side == 0 else c.width - 3
        ticks = max(0, (plane - game.x) / game.vx)
        span = c.height - 3
        unfolded = (game.y - 1 + ticks * game.vy) % (2 * span)
        target = 1 + min(unfolded, 2 * span - unfolded)
    error = target - game.paddles[side]
    return UP if error < -0.65 else DOWN if error > 0.65 else STAY


def tracker(game, side=0):
    """Strong reactive teacher; labels are fully observable in a single matrix.

    Tracking avoids asking an imitation learner to reconstruct subcell physics
    from quantized observations. The interception expert remains a benchmark.
    """
    grid = game.observation(side)[-1]
    ball_row = int(np.nonzero(grid == 3)[0][0])
    paddle_rows = np.nonzero(grid[:, 1] == 2)[0]
    center = float(paddle_rows.mean())
    error = ball_row - center
    return UP if error < -0.5 else DOWN if error > 0.5 else STAY
