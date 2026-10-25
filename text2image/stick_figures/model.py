import torch
from torch import nn
from text2image.stick_figures.data import VOCAB, MAX_TEXT, PAD


class PoseTransformer(nn.Module):
    """All output controls share a generic text encoder; no instruction parsing at inference."""
    def __init__(self, width=64, layers=2, heads=4):
        super().__init__()
        self.config = dict(width=width, layers=layers, heads=heads)
        self.embed = nn.Embedding(len(VOCAB) + 2, width, padding_idx=PAD)
        self.position = nn.Embedding(MAX_TEXT, width)
        layer = nn.TransformerEncoderLayer(width, heads, dim_feedforward=width * 4,
                                           dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.out = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 5))
        # TransformerEncoder clones a layer; initialize matrices independently.
        for name, parameter in self.encoder.named_parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(self, tokens):
        h = self.embed(tokens) + self.position.weight[:tokens.shape[1]]
        h = self.encoder(h, src_key_padding_mask=tokens == PAD)
        return self.out(self.norm(h[:, 0]))
