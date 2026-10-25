import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
LOG_FILE = os.path.join(HERE, "logs", "train.log")
CKPT_FILE = os.path.join(HERE, "checkpoints", "model.pt")

os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
os.makedirs(os.path.join(HERE, "checkpoints"), exist_ok=True)

from model import Model
from pipeline import MAX_LEN, VOCAB_SIZE, BOS, check_data, tokenizer_digest

BATCH_SIZE = 192
EVAL_INTERVAL = 10
LEARNING_RATE = 5e-4
WARMUP_STEPS = 4000
GRAD_CLIP = 1.0
EVAL_BATCH = 128

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_BF16 = DEVICE == "cuda"

if USE_BF16:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True



def load_pairs(name):
    src = np.fromfile(os.path.join(DATA_DIR, f"{name}_src.bin"), dtype=np.uint16)
    tgt = np.fromfile(os.path.join(DATA_DIR, f"{name}_tgt.bin"), dtype=np.uint16)
    n = src.size // MAX_LEN
    src = src.reshape(n, MAX_LEN)
    tgt = tgt.reshape(n, MAX_LEN)
    return torch.from_numpy(src.astype(np.int64)), torch.from_numpy(tgt.astype(np.int64))


def trim_batch(src, tgt):
    max_src_len = max(1, int((src != 0).sum(dim=1).max()))
    max_tgt_len = max(2, int((tgt != 0).sum(dim=1).max()) + 1)
    src = src[:, :max_src_len]
    tgt = tgt[:, :max_tgt_len]
    return src, tgt


def get_batch(src, tgt):
    pick = np.random.randint(0, src.shape[0], size=BATCH_SIZE)
    return src[pick].to(DEVICE), tgt[pick].to(DEVICE)


def ids_to_matchable(rows):
    """
    Mask only PAD (id 0). This matches what translate.py free-running decode
    expects: PAD is invisible; EOS is a legal token to predict.
    """
    rows = rows.clone()
    rows[rows == 0] = -100
    return rows




def learning_rate_at_step(step):
    """One-based absolute update number: linear warmup, then inverse-square-root decay."""
    if step < 1:
        raise ValueError("Training step must be at least 1")
    return LEARNING_RATE * min(step / WARMUP_STEPS, (WARMUP_STEPS / step) ** 0.5)


def schedule_config():
    return {"name": "inverse_sqrt", "peak_lr": LEARNING_RATE, "warmup_steps": WARMUP_STEPS}


def parse_args():
    parser = argparse.ArgumentParser(description="Train locally; --steps is the number of additional updates.")
    parser.add_argument("--resume", action="store_true", help="Continue checkpoints/model.pt and append to the log")
    parser.add_argument("--steps", type=int, default=5000, help="Updates to run this invocation (default: 5000)")
    parser.add_argument("--start-step", type=int, help="Completed updates for an older weights-only checkpoint")
    args = parser.parse_args()
    if args.steps <= 0 or (args.start_step is not None and args.start_step < 0):
        parser.error("--steps must be positive; --start-step must be nonnegative")
    return args


def restore_checkpoint(path, model, optimizer, digest, legacy_step=None):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format_version") != 2 or state.get("tokenizer_sha256") != digest:
        raise ValueError("Checkpoint does not match the current data/tokenizer.")
    if "lr_schedule" in state and state["lr_schedule"] != schedule_config():
        raise ValueError("Checkpoint LR schedule differs from the current training configuration.")
    full_resume = "optimizer" in state and "step" in state
    if not full_resume and legacy_step is None:
        raise ValueError("This checkpoint has no training step. Use --resume --start-step 5000 "
                         "if it completed 5000 updates; its optimizer will start fresh.")
    if full_resume and legacy_step is not None:
        raise ValueError("This checkpoint already stores its step; omit --start-step.")
    model.load_state_dict(state["model"])
    if not full_resume:
        print("Restored weights; fresh optimizer, LR follows the absolute training step.")
        return legacy_step, 0.0
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"])
    np_state = state["numpy_rng"]
    np.random.set_state((np_state[0], np.array(np_state[1], dtype=np.uint32),
                         np_state[2], np_state[3], np_state[4]))
    if torch.cuda.is_available() and "cuda_rng" in state:
        if len(state["cuda_rng"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    print(f"Restored model and optimizer at step {state['step']}.")
    return state["step"], state["elapsed_seconds"]


def save_checkpoint(path, model, optimizer, step, elapsed_seconds, digest):
    np_state = np.random.get_state()
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "format_version": 2, "tokenizer_sha256": digest, "step": step,
             "lr_schedule": schedule_config(), "elapsed_seconds": elapsed_seconds,
             "torch_rng": torch.get_rng_state(),
             "numpy_rng": (np_state[0], np_state[1].tolist(), *np_state[2:])}
    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
    # Replace only after the full checkpoint is written successfully.
    temporary = str(path) + ".tmp"
    torch.save(state, temporary)
    os.replace(temporary, path)


def main():
    args = parse_args()
    check_data(DATA_DIR)
    model = Model().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), eps=1e-8)
    start_step = 0
    elapsed_seconds = 0.0
    if args.resume:
        start_step, elapsed_seconds = restore_checkpoint(
            CKPT_FILE, model, optimizer, tokenizer_digest(DATA_DIR), args.start_step)
    elif args.start_step is not None:
        raise ValueError("--start-step requires --resume")
    print(f"training steps {start_step + 1} through {start_step + args.steps}")
    print(f"parameters {sum(p.numel() for p in model.parameters()):,}")
    print(f"device {DEVICE}")
    print(f"LR at next step: {learning_rate_at_step(start_step + 1):.8g} (absolute-step inverse sqrt)")

    train_src, train_tgt = load_pairs("train")
    val_src, val_tgt = load_pairs("val")

    log_exists = os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 0
    log = open(LOG_FILE, "a" if args.resume else "w")
    if not args.resume or not log_exists:
        log.write("step train_loss val_loss seconds\n")
    start_time = time.time()

    total_steps = start_step + args.steps

    for step in range(start_step, total_steps):
        lr = learning_rate_at_step(step + 1)
        optimizer.param_groups[0]["lr"] = lr
        src, tgt = get_batch(train_src, train_tgt)
        src, tgt = trim_batch(src, tgt)
        tgt_in = torch.cat([torch.full_like(tgt[:, :1], BOS), tgt[:, :-1]], dim=1)
        with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=USE_BF16):
            logits = model(src, tgt_in)
        target = ids_to_matchable(tgt)
        loss = F.cross_entropy(logits.float().reshape(-1, VOCAB_SIZE), target.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if (step + 1) % EVAL_INTERVAL == 0 or step + 1 == total_steps:
            model.eval()
            with torch.no_grad():
                losses = []
                for b in range(max(1, val_src.shape[0] // EVAL_BATCH)):
                    b_src = val_src[b * EVAL_BATCH : (b + 1) * EVAL_BATCH].to(DEVICE)
                    b_tgt = val_tgt[b * EVAL_BATCH : (b + 1) * EVAL_BATCH].to(DEVICE)
                    b_src, b_tgt = trim_batch(b_src, b_tgt)
                    with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=USE_BF16):
                        b_in = torch.cat([torch.full_like(b_tgt[:, :1], BOS), b_tgt[:, :-1]], dim=1)
                        logits = model(b_src, b_in)
                    target = ids_to_matchable(b_tgt)
                    losses.append(F.cross_entropy(logits.float().reshape(-1, VOCAB_SIZE), target.reshape(-1)).item())
                val_loss = sum(losses) / len(losses)
            model.train()
            line = f"{step + 1} {loss.item():.4f} {val_loss:.4f} {elapsed_seconds + time.time() - start_time:.0f}"
            print(line)
            log.write(line + "\n")
            log.flush()
            save_checkpoint(CKPT_FILE, model, optimizer, step + 1,
                        elapsed_seconds + time.time() - start_time, tokenizer_digest(DATA_DIR))

    log.close()


if __name__ == "__main__":
    main()
