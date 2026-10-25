import torch
import torch.nn as nn
import torch.nn.functional as F

from en2fr.pipeline import MAX_LEN, VOCAB_SIZE, PAD

D_MODEL = 256
N_HEADS = 8
N_LAYERS = 4
FF_DIM = 1024


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
        attn_out, _ = self.self_attn(h, h, h, attn_mask=attn_mask, key_padding_mask=src_pad_mask if memory is None else None, need_weights=False)
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
            h = layer(h, None, None, src == PAD)
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

