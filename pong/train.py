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
from .model import BOS, CHECKPOINT, HORIZON, PongTransformer
from .prepare_data import DATA


def unpack(matrices):
    shifts = torch.tensor([0, 2, 4, 6], device=matrices.device, dtype=torch.uint8)
    return ((matrices.unsqueeze(-1) >> shifts) & 3).flatten(-2)


def decoder_inputs(target):
    return torch.cat((torch.full_like(target[:, :1], BOS), target[:, :-1]), dim=1)


@torch.inference_mode()
def validate(model, x, y, batch_size=128):
    model.eval()
    correct, loss, exact = 0, 0.0, 0
    positions = torch.zeros(y.shape[1], device=y.device, dtype=torch.long)
    confusion = torch.zeros(3, 3, dtype=torch.long, device=y.device)
    for start in range(0, len(y), batch_size):
        target = y[start:start + batch_size]
        logits = model(x[start:start + batch_size], decoder_inputs(target))
        prediction = logits.argmax(-1)
        loss += F.cross_entropy(logits.flatten(0, 1), target.flatten(), reduction='sum').item()
        correct += (prediction == target).sum().item()
        exact += (prediction == target).all(-1).sum().item()
        positions += (prediction == target).sum(0)
        confusion += torch.bincount((target * 3 + prediction).flatten(), minlength=9).reshape(3, 3)
    generated, _ = model.generate(x[:min(256, len(y))])
    return dict(loss=loss / y.numel(), accuracy=correct / y.numel(),
                sequence_accuracy=exact / len(y), position_accuracy=(positions / len(y)).tolist(),
                generated_accuracy=(generated == y[:len(generated)]).float().mean().item(),
                generated_sequence_accuracy=(generated == y[:len(generated)]).all(-1).float().mean().item(),
                confusion=confusion.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--steps', type=int, default=6000, help='Additional optimizer updates')
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
    if not args.data.exists():
        parser.error('Record demonstrations first: python -m pong.prepare_data')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    with np.load(args.data, allow_pickle=False) as data:
        game_config = Config(**json.loads(str(data['config'])))
        teacher = str(data.get('teacher', 'intercept'))
        if int(data.get('horizon', 1)) != HORIZON:
            parser.error('Record ten-action sequences first: python -m pong.prepare_data')
        tx = torch.tensor(data['train_x'], device=device, dtype=torch.uint8)
        ty = torch.tensor(data['train_y'], device=device, dtype=torch.long)
        vx = torch.tensor(data['val_x'], device=device, dtype=torch.uint8)
        vy = torch.tensor(data['val_y'], device=device, dtype=torch.long)
        if bool(data.get('packed', False)):
            tx, vx = unpack(tx), unpack(vx)
    previous, best = 0, float('inf')
    state = None
    if args.resume:
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        if state['config'].get('horizon') != HORIZON:
            parser.error('Cannot resume a single-action checkpoint as a ten-action model')
        if state['game_config'] != asdict(game_config):
            parser.error('Dataset and checkpoint game configurations differ')
        if state.get('teacher', 'intercept') != teacher:
            parser.error('Dataset and checkpoint teachers differ; start a fresh model')
        model = PongTransformer(**state['config']).to(device)
        model.load_state_dict(state['model'])
        previous = state['steps']
        # A new dataset may have a different validation distribution.
        best = validate(model, vx, vy)['loss']
    else:
        model = PongTransformer(height=game_config.height, width=game_config.width,
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
            saved = dict(config=model.config, game_config=asdict(game_config), teacher=teacher, model=model.state_dict(),
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
