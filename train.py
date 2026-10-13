import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
LOG_FILE = os.path.join(HERE, "logs", "train.log")
CKPT_FILE = os.path.join(HERE, "checkpoints", "model.pt")

os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
os.makedirs(os.path.join(HERE, "checkpoints"), exist_ok=True)

MAX_LEN = 60
VOCAB_SIZE = 16_000
D_MODEL = 256
N_HEADS = 8
N_LAYERS = 4
FF_DIM = 1024

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


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(D_MODEL, N_HEADS, batch_first=True)
        self.ln_1 = nn.LayerNorm(D_MODEL)
        self.cross_attn = nn.MultiheadAttention(D_MODEL, N_HEADS, batch_first=True)
        self.ln_2 = nn.LayerNorm(D_MODEL)
        self.ff = nn.Linear(D_MODEL, FF_DIM)
        self.ff_out = nn.Linear(FF_DIM, D_MODEL)
        self.ln_3 = nn.LayerNorm(D_MODEL)

    def forward(self, h, attn_mask, memory, src_pad_mask):
        attn_out, _ = self.self_attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        h = self.ln_1(h + attn_out)
        if memory is not None:
            attn_out, _ = self.cross_attn(h, memory, memory, key_padding_mask=src_pad_mask, need_weights=False)
            h = self.ln_2(h + attn_out)
        return self.ln_3(h + self.ff_out(F.gelu(self.ff(h))))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, D_MODEL, padding_idx=0)
        self.pos = nn.Embedding(MAX_LEN, D_MODEL)
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        self.layers = nn.ModuleList([Layer() for _ in range(N_LAYERS)])

    def forward(self, src):
        length = src.shape[1]
        h = self.embed(src) + self.pos.weight[:length]
        for layer in self.layers:
            h = layer(h, None, None, None)
        return h


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, D_MODEL, padding_idx=0)
        self.pos = nn.Embedding(MAX_LEN, D_MODEL)
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        self.layers = nn.ModuleList([Layer() for _ in range(N_LAYERS)])
        self.out = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
        self.out.weight = self.embed.weight

    def forward(self, tgt_idx, memory, src_pad_mask, step=0):
        length = tgt_idx.shape[1]
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=tgt_idx.device), diagonal=1)
        h = self.embed(tgt_idx) + self.pos.weight[step : step + length]
        for layer in self.layers:
            h = layer(h, causal, memory, src_pad_mask)
        return self.out(h)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, src, tgt_in):
        src_pad_mask = src == 0
        memory = self.encoder(src)
        logits = self.decoder(tgt_in, memory, src_pad_mask)
        return logits


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
    tgt_in = tgt[:, :-1]
    with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=USE_BF16):
        logits = model(src, tgt_in)
    target = ids_to_matchable(tgt[:, 1:])
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
                    logits = model(b_src, b_tgt[:, :-1])
                target = ids_to_matchable(b_tgt[:, 1:])
                losses.append(F.cross_entropy(logits.float().reshape(-1, VOCAB_SIZE), target.reshape(-1)).item())
            val_loss = sum(losses) / len(losses)
        model.train()
        line = f"{step + 1} {loss.item():.4f} {val_loss:.4f} {time.time() - start_time:.0f}"
        print(line)
        log.write(line + "\n")
        log.flush()
        torch.save(model.state_dict(), CKPT_FILE)

log.close()
