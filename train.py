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
WARMUP_STEPS = 100
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




check_data(DATA_DIR)
model = Model().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), eps=1e-8)
print(f"parameters {sum(p.numel() for p in model.parameters()):,}")
print(f"device {DEVICE}")

train_src, train_tgt = load_pairs("train")
val_src, val_tgt = load_pairs("val")

log = open(LOG_FILE, "w")
log.write("step train_loss val_loss seconds\n")
start_time = time.time()

total_steps = 5000

for step in range(total_steps):
    lr = LEARNING_RATE * min(1.0, (step + 1) / WARMUP_STEPS)
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
        line = f"{step + 1} {loss.item():.4f} {val_loss:.4f} {time.time() - start_time:.0f}"
        print(line)
        log.write(line + "\n")
        log.flush()
        torch.save({"model": model.state_dict(), "format_version": 2,
                    "tokenizer_sha256": tokenizer_digest(DATA_DIR)}, CKPT_FILE)

log.close()
