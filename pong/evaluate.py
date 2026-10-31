"""Headless closed-loop matches, including two independent model checkpoints."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random

from .game import Config, Pong, expert, tracker


def controller(name, config=None):
    if name in ('expert', 'tracker', 'random', 'human'):
        return None, config or Config()
    from .model import ModelPlayer, load
    model, loaded_config = load(Path(name))
    if config is not None and config != loaded_config:
        raise ValueError('The two checkpoints use different game configurations')
    return ModelPlayer(model), loaded_config


def action(name, player, game, side, rng):
    if name == 'expert':
        return expert(game, side)
    if name == 'tracker':
        return tracker(game, side)
    if name == 'random':
        return rng.randrange(3)
    return player(game.observation(side) if player.needs_plan else None)


def match(left='expert', right='expert', games=5, ticks=3000, seed=10000000):
    if 'human' in (left, right):
        raise ValueError('Human controls require python -m pong.demo')
    if min(games, ticks) < 1:
        raise ValueError('games and ticks must be positive')
    lp, config = controller(left)
    rp, other = controller(right, config if lp else None)
    config = other if rp else config
    rng = random.Random(seed)
    totals, hits = [0, 0], [0, 0]
    wins, draws = [0, 0], 0
    for index in range(games):
        game = Pong(seed + index, config)
        for player in (lp, rp):
            if player is not None:
                player.reset()
        for _ in range(ticks):
            game.step(action(left, lp, game, 0, rng), action(right, rp, game, 1, rng))
        for side in range(2):
            totals[side] += game.score[side]
            hits[side] += game.hits[side]
        if game.score[0] == game.score[1]:
            draws += 1
        else:
            wins[int(game.score[1] > game.score[0])] += 1
    # Every opponent point is a missed return; serves also contribute misses.
    returns = [hits[s] / max(1, hits[s] + totals[1 - s]) for s in range(2)]
    return dict(left=left, right=right, games=games, ticks_per_game=ticks, seed=seed,
                points=totals, paddle_hits=hits, return_rate=returns, wins=wins, draws=draws,
                plan_refills=[player.refills if player else None for player in (lp, rp)],
                game_config=asdict(config))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left', default='pong/checkpoints/seq10.pt', help='Checkpoint, expert, tracker, or random')
    parser.add_argument('--right', default='expert', help='Checkpoint, expert, tracker, or random')
    parser.add_argument('--games', type=int, default=5)
    parser.add_argument('--ticks', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=10000000)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    result = match(args.left, args.right, args.games, args.ticks, args.seed)
    report = json.dumps(result, indent=2)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding='utf-8')


if __name__ == '__main__':
    main()
