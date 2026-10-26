"""Procedural perfect mazes and deterministic shortest-path labels."""
import random
from collections import deque

MOVES = ('UP', 'DOWN', 'LEFT', 'RIGHT', 'END', 'BOS', 'PAD')
DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))
END, BOS, PAD = 4, 5, 6


def solve(grid, start, goal):
    queue = deque([start])
    parents = {start: None}
    while queue:
        p = queue.popleft()
        if p == goal:
            path = []
            while parents[p] is not None:
                p, action = parents[p]
                path.append(action)
            return list(reversed(path)) + [END]
        for action, (dr, dc) in enumerate(DELTAS):
            q = p[0] + dr, p[1] + dc
            if 0 <= q[0] < len(grid) and 0 <= q[1] < len(grid) and grid[q[0]][q[1]] != 0 and q not in parents:
                parents[q] = p, action
                queue.append(q)
    raise ValueError('Unreachable goal')


def generate(seed, size=7):
    if size < 5 or size % 2 != 1:
        raise ValueError('Maze size must be odd and at least 5')
    rng = random.Random(seed)
    grid = [[0] * size for _ in range(size)]
    stack = [(1, 1)]
    grid[1][1] = 1
    while stack:
        r, c = stack[-1]
        options = [(r+2*dr, c+2*dc, dr, dc) for dr, dc in DELTAS
                   if 0 < r+2*dr < size-1 and 0 < c+2*dc < size-1 and grid[r+2*dr][c+2*dc] == 0]
        if options:
            nr, nc, dr, dc = rng.choice(options)
            grid[r+dr][c+dc] = grid[nr][nc] = 1
            stack.append((nr, nc))
        else:
            stack.pop()
    cells = [(r, c) for r in range(size) for c in range(size) if grid[r][c]]
    start, goal = rng.sample(cells, 2)
    grid[start[0]][start[1]] = 2
    grid[goal[0]][goal[1]] = 3
    return grid, start, goal, solve(grid, start, goal)


def advance(grid, position, action):
    if action >= 4:
        return position, False
    dr, dc = DELTAS[action]
    r, c = position[0]+dr, position[1]+dc
    valid = 0 <= r < len(grid) and 0 <= c < len(grid) and grid[r][c] != 0
    return ((r, c) if valid else position), valid
