"""Measure actual open-loop Snake play on unseen seeded episodes."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random

import numpy as np
import torch

from .game import Config, Snake, expert
from .model import CHECKPOINT, ModelPlayer, load


def controller(name, device='cpu'):
    if str(name) in ('expert', 'random', 'human'):
        return None, Config()
    model, config = load(name, device)
    return ModelPlayer(model), config


def action(name, player, game, rng):
    if player is not None:
        return player(game.observation() if player.needs_plan else None)
    return expert(game) if str(name) == 'expert' else rng.randrange(3)


def evaluate(name=CHECKPOINT, games=30, ticks=500, seed=10000000):
    if games < 1 or ticks < 1 or str(name) == 'human':
        raise ValueError('Positive game/tick counts and an automated player are required')
    player, config = controller(name)
    results = []
    for index in range(games):
        game = Snake(seed + index, config)
        rng = random.Random(seed + index)
        if player:
            player.reset()
        before = player.refills if player else 0
        for _ in range(ticks):
            move = action(name, player, game, rng)
            destination, _ = game.destination(move)
            game.step(move)
            if not game.alive or game.won:
                break
        wall = not (0 < destination[0] < config.height - 1 and 0 < destination[1] < config.width - 1)
        results.append(dict(seed=seed + index, score=game.score, ticks=game.ticks,
                            died=not game.alive, won=game.won,
                            wall_death=not game.alive and wall,
                            refills=player.refills - before if player else 0))
    return dict(player=str(name), games=games, max_ticks=ticks, seed=seed,
                board=asdict(config), geometry_constrained=player is not None,
                mean_score=float(np.mean([r['score'] for r in results])),
                mean_ticks=float(np.mean([r['ticks'] for r in results])),
                deaths=sum(r['died'] for r in results),
                wall_deaths=sum(r['wall_death'] for r in results), episodes=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--player', default=str(CHECKPOINT))
    parser.add_argument('--games', type=int, default=30)
    parser.add_argument('--ticks', type=int, default=500)
    parser.add_argument('--seed', type=int, default=10000000)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if min(args.games, args.ticks) < 1 or args.player == 'human':
        parser.error('Use positive counts and an automated player')
    torch.set_num_threads(2)
    try:
        report = evaluate(args.player, args.games, args.ticks, args.seed)
    except FileNotFoundError:
        parser.error('Train first: python -m snake.prepare_data; python -m snake.train')
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
