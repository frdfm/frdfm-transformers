import argparse
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from text2image.model import ImageTransformer, CLASSES, TEMPLATES, PREFIX, encode_prompt
from text2image.generate import save_grid

HERE = Path(__file__).resolve().parent


def load_data(split, side=28):
    folder = HERE / "data" if side == 14 else HERE / "data" / str(side)
    path = folder / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Prepare data first: python -m text2image.prepare_data --size {side}")
    with np.load(path) as data:
        if data["pixels"].shape[1] != side * side:
            raise ValueError("Data resolution does not match --size")
        return torch.from_numpy(data["pixels"].astype(np.int64)), data["labels"].tolist()


def make_captions(labels, randomize=True):
    choices = torch.randint(len(TEMPLATES), (len(labels),)).tolist() if randomize else [0] * len(labels)
    return torch.tensor([encode_prompt(TEMPLATES[t].format(CLASSES[label]))
                         for label, t in zip(labels, choices)], dtype=torch.long)


def pixel_loss(model, captions, pixels):
    # Last caption position predicts pixel 0; each later position predicts the next pixel.
    inputs = torch.cat([captions, pixels[:, :-1]], dim=1)
    logits = model(inputs)[:, PREFIX - 1:]
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), pixels.reshape(-1))


@torch.inference_mode()
def evaluate(model, pixels, captions, device, batch_size, batches):
    model.eval()
    total, count = 0.0, 0
    limit = len(pixels) if batches == 0 else min(len(pixels), batches * batch_size)
    for start in range(0, limit, batch_size):
        x, text = pixels[start:start + batch_size].to(device), captions[start:start + batch_size].to(device)
        total += pixel_loss(model, text, x).item() * len(x)
        count += len(x)
    model.train()
    return total / count


def main():
    parser = argparse.ArgumentParser(description="Train a tiny caption-to-pixels Transformer from scratch.")
    parser.add_argument("--size", type=int, choices=[14, 28], default=28)
    parser.add_argument("--steps", type=int, default=5000, help="Additional optimizer updates")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--val-batches", type=int, default=0, help="0 evaluates all validation images")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-samples", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.eval_every) < 1 or args.val_batches < 0:
        parser.error("Steps, batch size, and eval interval must be positive; val batches must be nonnegative")
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    x, labels = load_data("train", args.size)
    val_x, val_labels = load_data("val", args.size)
    val_text = make_captions(val_labels, randomize=False)
    model = ImageTransformer(side=args.size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    run_suffix = Path() if args.size == 14 else Path(str(args.size))
    checkpoint = HERE / "checkpoints" / run_suffix / "model.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    logs = HERE / "logs" / run_suffix
    logs.mkdir(parents=True, exist_ok=True)
    start_step = 0
    best = math.inf
    if args.resume:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step, best = state["step"], state["best_val"]
        torch.set_rng_state(state["rng"])
        if device == "cuda" and "cuda_rng" in state:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters on {device}; steps {start_step + 1}-{start_step + args.steps}")
    started = time.time()
    logpath = logs / "train.csv"
    needs_header = not args.resume or not logpath.exists()
    with logpath.open("a" if args.resume else "w") as log:
        if needs_header:
            log.write("step,train_loss,val_loss,lr,run_seconds\n")
        model.train()
        for step in range(start_step + 1, start_step + args.steps + 1):
            lr = 3e-4 * min(step / 500, (500 / step) ** 0.5)
            for group in optimizer.param_groups:
                group["lr"] = lr
            indices = torch.randint(len(x), (args.batch_size,))
            pixels = x[indices].to(device)
            captions = make_captions([labels[i] for i in indices.tolist()]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = pixel_loss(model, captions, pixels)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss; checkpoint was not overwritten")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step % args.eval_every == 0 or step == start_step + args.steps:
                val = evaluate(model, val_x, val_text, device, args.batch_size, args.val_batches)
                improved = val < best
                best = min(best, val)
                elapsed = time.time() - started
                print(f"step {step} train {loss.item():.4f} val {val:.4f} lr {lr:.2g} seconds {elapsed:.0f}", flush=True)
                log.write(f"{step},{loss.item():.6f},{val:.6f},{lr:.8g},{elapsed:.1f}\n")
                log.flush()
                if not args.no_samples:
                    # Preview sampling must not change the subsequent training RNG sequence.
                    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device == "cuda" else []):
                        torch.manual_seed(123)
                        text = make_captions(list(range(10)), randomize=False).to(device)
                        model.eval()
                        samples = model.sample(text)
                        save_grid(samples, HERE / "samples" / run_suffix / f"step-{step:06d}.png", columns=5, labels=CLASSES)
                        model.train()
                state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                             config=model.config, step=step, best_val=best, rng=torch.get_rng_state())
                if device == "cuda":
                    state["cuda_rng"] = torch.cuda.get_rng_state_all()
                temp = checkpoint.with_suffix(".tmp")
                torch.save(state, temp)
                temp.replace(checkpoint)
                if improved:
                    torch.save(state, checkpoint.with_name("best.pt"))
                if device == "cuda":
                    print(f"peak GPU allocated: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
