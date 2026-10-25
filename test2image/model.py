"""A small causal Transformer: caption tokens followed by grayscale pixel tokens."""
import re
import torch
from torch import nn

SIDE, LEVELS, PREFIX = 14, 16, 10
PIXELS = SIDE * SIDE
CLASSES = ("t-shirt", "trousers", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot")
TEMPLATES = ("draw a {}", "a drawing of a {}", "a grayscale image of a {}", "{}")
WORDS = sorted(set(re.findall(r"[a-z]+", " ".join(t.format(c) for t in TEMPLATES for c in CLASSES))))
PAD = LEVELS
WORD_IDS = {word: LEVELS + 1 + i for i, word in enumerate(WORDS)}
VOCAB = LEVELS + 1 + len(WORDS)


def encode_prompt(prompt):
    words = re.findall(r"[a-z]+", prompt.lower())
    if not words or len(words) > PREFIX:
        raise ValueError(f"Use a short clothing prompt (at most {PREFIX} words).")
    unknown = set(words) - WORD_IDS.keys()
    if unknown:
        raise ValueError(f"Unknown words: {', '.join(sorted(unknown))}. Try 'draw a bag'.")
    ids = [WORD_IDS[w] for w in words]
    return ids + [PAD] * (PREFIX - len(ids))


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0.1, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(),
                                nn.Linear(width * 4, width), nn.Dropout(0.1))

    def forward(self, x, causal, padding):
        h = self.norm1(x)
        x = x + self.attention(h, h, h, attn_mask=causal,
                               key_padding_mask=padding, need_weights=False)[0]
        return x + self.ff(self.norm2(x))


class ImageTransformer(nn.Module):
    def __init__(self, width=128, layers=4, heads=4, side=14):
        super().__init__()
        if side not in (14, 28):
            raise ValueError("Supported image sizes are 14 and 28")
        self.side = side
        self.pixels = side * side
        self.config = dict(width=width, layers=layers, heads=heads, side=side)
        self.embed = nn.Embedding(VOCAB, width)
        self.position = nn.Embedding(PREFIX + self.pixels, width)
        self.blocks = nn.ModuleList([Block(width, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, LEVELS)

    def forward(self, tokens):
        length = tokens.shape[1]
        causal = torch.ones(length, length, device=tokens.device, dtype=torch.bool).triu(1)
        x = self.embed(tokens) + self.position.weight[:length]
        for block in self.blocks:
            x = block(x, causal, tokens == PAD)
        return self.out(self.norm(x))

    @torch.inference_mode()
    def sample(self, captions, temperature=0.8):
        if temperature <= 0:
            raise ValueError("Temperature must be positive.")
        tokens = captions
        for _ in range(self.pixels):
            logits = self(tokens)[:, -1] / temperature
            pixel = torch.multinomial(logits.softmax(-1), 1)
            tokens = torch.cat([tokens, pixel], dim=1)
        return tokens[:, PREFIX:].reshape(-1, self.side, self.side)
