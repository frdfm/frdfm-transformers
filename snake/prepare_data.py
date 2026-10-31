"""Record food-seeking plans followed by safe movement after the known meal."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import random

import numpy as np

from .game import Config, Snake, expert

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'seq10-wide-v2.npz'
HORIZON = 10
TEACHER = 'bfs-safe-continuation-v2'


def future_actions(game):
    future = deepcopy(game)
    targets = np.full(HORIZON, -100, dtype=np.int64)
    for tick in range(HORIZON):
        move = expert(future)
        # Never teach a move into an unavoidable collision.
        if not future.safe(move):
            break
        targets[tick] = move
        ate = future.step(move)
        # Do not chase the next random food: plan safe movement for the rest
        # of this queue using only geometry visible at planning time.
        if ate:
            future.food = None
        if not future.alive or future.won:
            break
    return targets


def record(count, seed, config=None):
    config = config or Config()
    rng = random.Random(seed)
    x = np.empty((count, config.history, config.height, config.width), dtype=np.uint8)
    y = np.empty((count, HORIZON), dtype=np.int64)
    episodes = np.empty(count, dtype=np.int64)
    collected, episode = 0, 0
    while collected < count:
        game = Snake(seed + episode, config)
        # Deliberately include corners and edges, not only uniform food starts.
        if rng.random() < 0.4:
            edge = rng.choice((1, config.height - 2))
            col = rng.choice((1, config.width - 2)) if rng.random() < 0.5 else rng.randrange(1, config.width - 1)
            if (edge, col) not in game.body:
                game.food = (edge, col)
                game.frames.clear()
                game.frames.extend(game.matrix() for _ in range(config.history))
        for tick in range(500):
            if tick % 3 == 0:
                target = future_actions(game)
                if target[0] != -100:
                    x[collected], y[collected], episodes[collected] = game.observation(), target, episode
                    collected += 1
                    if collected == count:
                        break
            move = expert(game)
            if rng.random() < 0.12:
                safe = [a for a in range(3) if game.safe(a)]
                move = rng.choice(safe) if safe else move
            game.step(move)
            if not game.alive or game.won:
                break
        episode += 1
    return x, y, episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=20000)
    parser.add_argument('--validation-samples', type=int, default=1500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=DATA)
    args = parser.parse_args()
    if min(args.samples, args.validation_samples) < 1:
        parser.error('Sample counts must be positive')
    config = Config()
    print('Recording training episodes...', flush=True)
    tx, ty, te = record(args.samples, args.seed, config)
    # Each episode contributes at least one example, so seed ranges cannot overlap.
    validation_seed = args.seed + args.samples + 1000000
    print('Recording held-out episodes...', flush=True)
    vx, vy, ve = record(args.validation_samples, validation_seed, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, train_x=tx, train_y=ty, train_episode=te,
                            val_x=vx, val_y=vy, val_episode=ve, seed=args.seed,
                            validation_seed=validation_seed, config=json.dumps(asdict(config)),
                            horizon=HORIZON, teacher=TEACHER)
    temporary.replace(args.output)
    print(f'Saved {len(ty):,} training / {len(vy):,} validation sequences to {args.output}')
    print('Supervised tokens by future position:', (ty != -100).sum(0).tolist())


if __name__ == '__main__':
    main()
