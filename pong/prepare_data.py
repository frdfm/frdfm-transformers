"""Record expert labels on varied rallies, optionally including model rollouts."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import random

import numpy as np

from .game import Config, Pong, expert, tracker

DATA = Path(__file__).resolve().parent / 'data' / 'seq10.npz'
HORIZON = 10


def future_actions(game, policy, horizon=HORIZON):
    """Counterfactual expert rollout from this state, without changing the game.

    Skip targets crossing a serve: the fresh random serve is not observable yet.
    """
    future = deepcopy(game)
    targets = [[], []]
    for tick in range(horizon):
        moves = [policy(future, side) for side in range(2)]
        for side in range(2):
            targets[side].append(moves[side])
        if future.step(*moves) is not None and tick < horizon - 1:
            return None
    return targets


def pack(matrices):
    """Four 2-bit categorical cells per byte, losslessly."""
    if matrices.shape[-1] % 4:
        raise ValueError('Packed matrix width must be divisible by four')
    return (matrices[..., 0::4] | matrices[..., 1::4] << 2 |
            matrices[..., 2::4] << 4 | matrices[..., 3::4] << 6)


def record(count, seed, config=None, player=None, packed=False, teacher='tracker'):
    config = config or Config()
    rng = random.Random(seed)
    shape = (count, config.history, config.height, config.width // 4 if packed else config.width)
    observations = np.empty(shape, dtype=np.uint8)
    actions, episodes = [], []
    players = [player, type(player)(player.model)] if player is not None else [None, None]
    episode = 0
    while len(actions) < count:
        game = Pong(seed + episode, config)
        model_sides = [player is not None and rng.random() < 0.5 for _ in range(2)]
        for model_player in players:
            if model_player is not None:
                model_player.reset()
        # Reachable randomized starts avoid a dataset dominated by centered serves.
        low = 1 + (config.paddle_height - 1) / 2
        high = config.height - 2 - (config.paddle_height - 1) / 2
        game.paddles = [rng.uniform(low, high) for _ in range(2)]
        game.frames.clear()
        for _ in range(config.history):
            game.frames.append(game.matrix())
        for tick in range(256):
            policy = tracker if teacher == 'tracker' else expert
            labels = [policy(game, side) for side in range(2)]
            targets = future_actions(game, policy) if tick >= config.history and tick % 4 == 0 else None
            if targets is not None:
                for side in range(2):
                    observation = game.observation(side)
                    observations[len(actions)] = pack(observation) if packed else observation
                    actions.append(targets[side])
                    episodes.append(episode)
                    if len(actions) == count:
                        break
            moves = labels.copy()
            for side in range(2):
                if model_sides[side]:
                    learner = players[side]
                    moves[side] = learner(game.observation(side) if learner.needs_plan else None)
                elif rng.random() < 0.2:
                    moves[side] = rng.randrange(3)
            scored = game.step(*moves)
            if scored is not None or len(actions) == count:
                break
        episode += 1
    return observations, np.asarray(actions, dtype=np.int64), np.asarray(episodes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=30000)
    parser.add_argument('--validation-samples', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=DATA)
    parser.add_argument('--checkpoint', type=Path, help='DAgger: label states visited by this model')
    parser.add_argument('--append', action='store_true', help='Aggregate training decisions; preserve existing validation')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--teacher', choices=['tracker', 'intercept'], default='tracker')
    args = parser.parse_args()
    if min(args.samples, args.validation_samples) < 1:
        parser.error('Sample counts must be positive')
    config, player = Config(), None
    if args.checkpoint:
        import torch
        from .model import load, ModelPlayer
        torch.set_num_threads(4)
        device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
        model, config = load(args.checkpoint, device)
        player = ModelPlayer(model)
    existing = None
    if args.append:
        if not args.output.exists():
            parser.error('--append requires an existing dataset')
        with np.load(args.output, allow_pickle=False) as data:
            existing = {key: data[key] for key in data.files}
        if not bool(existing.get('packed', False)):
            for key in ('train_x', 'val_x'):
                existing[key] = pack(existing[key])
        if json.loads(str(existing['config'])) != asdict(config):
            parser.error('Existing dataset and model configurations differ')
        if str(existing.get('teacher', 'intercept')) != args.teacher:
            parser.error('Cannot append labels from a different teacher')
        if int(existing.get('horizon', 1)) != HORIZON:
            parser.error('Cannot append sequences to an old single-action dataset')
        # Every rally consumes at least one seed; counts give conservative bounds.
        ranges = existing.get('train_seed_ranges', np.asarray([
            [int(existing['seed']), int(existing['seed']) + len(existing['train_y'])]]))
        if any(args.seed <= end and args.seed + args.samples >= start for start, end in ranges):
            parser.error('Choose a distinct --seed when appending (for example 20000000)')
        validation_start = int(existing['validation_seed'])
        if args.seed <= validation_start + len(existing['val_y']) and args.seed + args.samples >= validation_start:
            parser.error('Append seed range overlaps held-out rally seeds')
    print('Recording training rallies...', flush=True)
    train_x, train_y, train_episode = record(args.samples, args.seed, config, player, packed=True, teacher=args.teacher)
    # Disjoint simulation seeds; no adjacent frames from a rally cross the split.
    if existing is None:
        validation_seed = args.seed + args.samples + 1000000
        print('Recording held-out expert rallies...', flush=True)
        val_x, val_y, val_episode = record(args.validation_samples, validation_seed, config, packed=True, teacher=args.teacher)
        training_seed = args.seed
        seed_ranges = np.asarray([[args.seed, args.seed + args.samples]], dtype=np.int64)
    else:
        validation_seed = int(existing['validation_seed'])
        training_seed = min(args.seed, int(existing['seed']))
        seed_ranges = np.concatenate((ranges, [[args.seed, args.seed + args.samples]]))
        train_episode += int(existing['train_episode'].max()) + 1
        train_x = np.concatenate((existing['train_x'], train_x))
        train_y = np.concatenate((existing['train_y'], train_y))
        train_episode = np.concatenate((existing['train_episode'], train_episode))
        val_x, val_y, val_episode = (existing[key] for key in ('val_x', 'val_y', 'val_episode'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, train_x=train_x, train_y=train_y,
                            train_episode=train_episode, val_x=val_x, val_y=val_y,
                            val_episode=val_episode, config=json.dumps(asdict(config)),
                            seed=training_seed, train_seed_ranges=seed_ranges, validation_seed=validation_seed,
                            packed=True, teacher=args.teacher, horizon=HORIZON)
    temporary.replace(args.output)
    print(f'Saved {len(train_y):,} training / {len(val_y):,} validation decisions to {args.output}')
    print('Training token counts [stay, up, down]:', np.bincount(train_y.flatten(), minlength=3).tolist())


if __name__ == '__main__':
    main()
