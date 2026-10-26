"""Train on procedural mazes; evaluate autoregressive routes on held-out mazes."""
import argparse
import json
from pathlib import Path
import random
import time
import torch
from torch.nn import functional as F
from .data import generate, advance, BOS, END, PAD
from .model import MazeTransformer, load

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / 'checkpoints' / 'model.pt'


def checkpoint_for(size):
    return CHECKPOINT if size == 7 else ROOT / 'checkpoints' / f'model-{size}.pt'


@torch.inference_mode()
def evaluate(model, count=100, min_route=0):
    device = next(model.parameters()).device
    solved = shortest = collisions = 0
    long_count = long_solved = 0
    seed = 1000000
    for _ in range(count):
        for attempt in range(10000):
            grid, start, goal, target = generate(seed, model.config['size'])
            seed += 1
            if len(target)-1 >= min_route:
                break
        else:
            raise ValueError('Could not find a maze meeting --min-route')
        is_long = len(target)-1 >= 24
        long_count += is_long
        memory = model.encode(torch.tensor([grid], device=device))
        tokens = [BOS]
        pos = start
        for _ in range(model.config['size']**2):
            action = model.decode(torch.tensor([tokens], device=device), memory)[0, -1].argmax().item()
            tokens.append(action)
            if action == END:
                solved += pos == goal
                long_solved += is_long and pos == goal
                shortest += pos == goal and len(tokens)-1 == len(target)
                break
            pos, valid = advance(grid, pos, action)
            if not valid:
                collisions += 1
                break
    return dict(mazes=count, min_route=min_route, solved=solved, shortest=shortest, collisions=collisions,
                routes_at_least_24_moves=long_count, long_routes_solved=long_solved)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--size', type=int, default=7)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate, including on resume')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--eval-count', type=int, default=100)
    parser.add_argument('--min-route', type=int, default=0, help='Minimum solution length for evaluation')
    args = parser.parse_args()
    checkpoint = checkpoint_for(args.size)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load(checkpoint, device) if args.resume or args.evaluate else MazeTransformer(args.size).to(device)
    if args.evaluate:
        print(json.dumps(evaluate(model, args.eval_count, args.min_route)), flush=True)
        return
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr or 0.0005)
    previous = 0
    if args.resume:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        optimizer.load_state_dict(state['optimizer'])
        previous = state['steps']
        torch.set_rng_state(state['rng'].cpu())
    if args.lr is not None:
        for group in optimizer.param_groups:
            group['lr'] = args.lr
    rng = random.Random(42+previous)
    # Finite pool keeps teacher labels inspectable and generation overhead low.
    pool = [generate(i, model.config['size']) for i in range(12000)]
    started = time.monotonic()
    model.train()
    print(f'Training on {device}; {sum(p.numel() for p in model.parameters()):,} parameters', flush=True)
    for step in range(1, args.steps+1):
        batch = rng.choices(pool, k=args.batch_size)
        grids = torch.tensor([b[0] for b in batch], device=device)
        length = max(len(b[3]) for b in batch)
        target = torch.full((len(batch), length), PAD, device=device, dtype=torch.long)
        for row, b in enumerate(batch):
            target[row, :len(b[3])] = torch.tensor(b[3], device=device)
        inputs = torch.cat((torch.full((len(batch), 1), BOS, device=device), target[:, :-1]), dim=1)
        logits = model(grids, inputs)
        loss = F.cross_entropy(logits.flatten(0, 1), target.flatten(), ignore_index=PAD)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            print(f'step {previous+step} loss {loss.item():.4f} elapsed {time.monotonic()-started:.0f}s', flush=True)
        if step % 500 == 0 or step == args.steps:
            checkpoint.parent.mkdir(exist_ok=True)
            temporary = checkpoint.with_suffix('.tmp')
            torch.save(dict(config=model.config, model=model.state_dict(), optimizer=optimizer.state_dict(),
                            steps=previous+step, rng=torch.get_rng_state()), temporary)
            temporary.replace(checkpoint)
    model.eval()
    report = evaluate(model, args.eval_count, args.min_route)
    (checkpoint.parent / f'evaluation-{args.size}.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
