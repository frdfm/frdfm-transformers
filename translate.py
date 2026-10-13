import os
import sys

import numpy as np
import torch
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_FILE = os.path.join(HERE, "checkpoints", "model.pt")
TOKENIZER_FILE = os.path.join(HERE, "data", "tokenizer.json")

MAX_LEN = 60
VOCAB_SIZE = 16_000
D_MODEL = 256
N_HEADS = 8
N_LAYERS = 4
FF_DIM = 1024

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PAD = 0
EOS = 1


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = torch.nn.MultiheadAttention(D_MODEL, N_HEADS, batch_first=True)
        self.ln_1 = torch.nn.LayerNorm(D_MODEL)
        self.cross_attn = torch.nn.MultiheadAttention(D_MODEL, N_HEADS, batch_first=True)
        self.ln_2 = torch.nn.LayerNorm(D_MODEL)
        self.ff = torch.nn.Linear(D_MODEL, FF_DIM)
        self.ff_out = torch.nn.Linear(FF_DIM, D_MODEL)
        self.ln_3 = torch.nn.LayerNorm(D_MODEL)

    def forward(self, h, attn_mask, memory, src_pad_mask):
        attn_out, _ = self.self_attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        h = self.ln_1(h + attn_out)
        if memory is not None:
            attn_out, _ = self.cross_attn(h, memory, memory, key_padding_mask=src_pad_mask, need_weights=False)
            h = self.ln_2(h + attn_out)
        return self.ln_3(h + self.ff_out(torch.nn.functional.gelu(self.ff(h))))


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB_SIZE, D_MODEL, padding_idx=0)
        self.pos = torch.nn.Embedding(MAX_LEN, D_MODEL)
        torch.nn.init.normal_(self.embed.weight, std=0.02)
        torch.nn.init.normal_(self.pos.weight, std=0.02)
        self.layers = torch.nn.ModuleList([Layer() for _ in range(N_LAYERS)])

    def forward(self, src):
        length = src.shape[1]
        h = self.embed(src) + self.pos.weight[:length]
        for layer in self.layers:
            h = layer(h, None, None, None)
        return h


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB_SIZE, D_MODEL, padding_idx=0)
        self.pos = torch.nn.Embedding(MAX_LEN, D_MODEL)
        torch.nn.init.normal_(self.embed.weight, std=0.02)
        torch.nn.init.normal_(self.pos.weight, std=0.02)
        self.layers = torch.nn.ModuleList([Layer() for _ in range(N_LAYERS)])
        self.out = torch.nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
        self.out.weight = self.embed.weight

    def forward(self, tgt_idx, memory, src_pad_mask, step=0):
        length = tgt_idx.shape[1]
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=tgt_idx.device), diagonal=1)
        h = self.embed(tgt_idx) + self.pos.weight[step : step + length]
        for layer in self.layers:
            h = layer(h, causal, memory, src_pad_mask)
        return self.out(h)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()


sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if not os.path.exists(CKPT_FILE):
    sys.exit(f"no checkpoint at {CKPT_FILE} - run train.py first")
if not os.path.exists(TOKENIZER_FILE):
    sys.exit(f"no tokenizer at {TOKENIZER_FILE} - run prepare_data.py first")
prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "the weather is nice today"
tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
src_ids = tokenizer.encode(prompt).ids[: MAX_LEN - 1]
sentence = torch.tensor([src_ids], dtype=torch.long, device=DEVICE)
src_pad_len = max(1, len(src_ids))  # pad to input length only, like trim_batch during training
pad = torch.tensor([[PAD] * (src_pad_len - len(src_ids))], dtype=torch.long, device=DEVICE)
src = torch.cat([sentence, pad], dim=1)

state = torch.load(CKPT_FILE, map_location=DEVICE, weights_only=True)
model = Model().to(DEVICE)
model.load_state_dict(state)
model.eval()

with torch.no_grad():
    memory = model.encoder(src)
    src_pad_mask = src == 0
    outputs = [EOS]
    for step in range(MAX_LEN):
        tgt = torch.tensor([outputs], dtype=torch.long, device=DEVICE)
        logits = model.decoder(tgt, memory, src_pad_mask)
        next_token = logits[:, -1].argmax(dim=-1).item()
        if next_token == EOS:
            break
        outputs.append(next_token)

print(tokenizer.decode(outputs[1:]))
