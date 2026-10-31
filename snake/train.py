"""Behavior cloning from recorded matrix histories and expert actions."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .game import Config
from .model import BOS, CHECKPOINT, HORIZON, SnakeTransformer
from .prepare_data import DATA


def decoder_inputs(target):
    return torch.cat((torch.full_like(target[:, :1], BOS), target[:, :-1].clamp_min(0)), dim=1)


@torch.inference_mode()
def validate(model, x, y, batch_size=128):
    model.eval()
    loss, total, correct, exact = 0.0, 0, 0, 0
    positions = torch.zeros(y.shape[1], device=y.device)
    counts = torch.zeros_like(positions)
    for start in range(0, len(y), batch_size):
        target = y[start:start + batch_size]
        valid = target != -100
        logits = model(x[start:start + batch_size], decoder_inputs(target))
        prediction = logits.argmax(-1)
        loss += F.cross_entropy(logits.flatten(0, 1), target.flatten(), reduction='sum').item()
        total += valid.sum().item()
        hits = (prediction == target) & valid
        correct += hits.sum().item()
        exact += (hits | ~valid).all(-1).sum().item()
        positions += hits.sum(0)
        counts += valid.sum(0)
    generated, _ = model.generate(x[:min(256, len(y))])
    target = y[:len(generated)]
    valid = target != -100
    hits = generated == target
    return dict(loss=loss / total, accuracy=correct / total,
                sequence_accuracy=exact / len(y), position_accuracy=(positions / counts.clamp_min(1)).tolist(),
                position_counts=counts.long().tolist(),
                generated_accuracy=((hits & valid).sum() / valid.sum()).item(),
                generated_sequence_accuracy=(hits | ~valid).all(-1).float().mean().item())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--steps', type=int, default=9000, help='Additional optimizer updates')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--eval-every', type=int, default=250)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.eval_every) < 1:
        parser.error('Steps, batch size and evaluation interval must be positive')
    if args.lr is not None and args.lr <= 0:
        parser.error('Learning rate must be positive')
    if not args.data.exists():
        parser.error('Record demonstrations first: python -m snake.prepare_data')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    with np.load(args.data, allow_pickle=False) as data:
        game_config = Config(**json.loads(str(data['config'])))
        teacher = str(data['teacher'])
        if int(data.get('horizon', 1)) != HORIZON:
            parser.error('Record ten-action sequences first: python -m snake.prepare_data')
        tx = torch.tensor(data['train_x'], device=device, dtype=torch.uint8)
        ty = torch.tensor(data['train_y'], device=device, dtype=torch.long)
        vx = torch.tensor(data['val_x'], device=device, dtype=torch.uint8)
        vy = torch.tensor(data['val_y'], device=device, dtype=torch.long)
    previous, best = 0, float('inf')
    state = None
    if args.resume:
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        if state.get('experiment') != 'snake-seq10' or state['config'].get('horizon') != HORIZON:
            parser.error('Resume requires a Snake ten-action checkpoint')
        if state['game_config'] != asdict(game_config):
            parser.error('Dataset and checkpoint game configurations differ')
        if state.get('teacher') != teacher:
            parser.error('Dataset and checkpoint teachers differ; start a fresh model')
        model = SnakeTransformer(**state['config']).to(device)
        model.load_state_dict(state['model'])
        previous = state['steps']
        best = min(state.get('best_loss', float('inf')), validate(model, vx, vy)['loss'])
    else:
        model = SnakeTransformer(height=game_config.height, width=game_config.width,
                                history=game_config.history, dim=args.dim, layers=args.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr or 0.0005)
    if state:
        optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['rng'].cpu())
        if device == 'cuda' and 'cuda_rng' in state:
            torch.cuda.set_rng_state(state['cuda_rng'].cpu())
        if args.lr is not None:
            for group in optimizer.param_groups:
                group['lr'] = args.lr
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    print(f'{device}: {sum(p.numel() for p in model.parameters()):,} parameters; '
          f'{len(ty):,} expert decisions', flush=True)
    for step in range(1, args.steps + 1):
        model.train()
        indices = torch.randint(len(ty), (args.batch_size,), device=device)
        target = ty[indices]
        loss = F.cross_entropy(model(tx[indices], decoder_inputs(target)).flatten(0, 1), target.flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            print(f'step {previous + step} loss {loss.item():.4f} '
                  f'elapsed {time.monotonic() - started:.0f}s', flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            report = validate(model, vx, vy)
            improved = report['loss'] < best
            best = min(best, report['loss'])
            saved = dict(experiment='snake-seq10', config=model.config, game_config=asdict(game_config), teacher=teacher, model=model.state_dict(),
                         optimizer=optimizer.state_dict(), steps=previous + step,
                         rng=torch.get_rng_state(), best_loss=best, validation=report)
            if device == 'cuda':
                saved['cuda_rng'] = torch.cuda.get_rng_state()
            temporary = args.checkpoint.with_suffix('.tmp')
            torch.save(saved, temporary)
            temporary.replace(args.checkpoint)
            if improved:
                torch.save(saved, args.checkpoint.with_name(args.checkpoint.stem + '-best.pt'))
            args.checkpoint.with_suffix('.json').write_text(json.dumps(
                dict(steps=previous + step, validation=report), indent=2), encoding='utf-8')
            print(f'validation {json.dumps(report)}', flush=True)


if __name__ == '__main__':
    main()
