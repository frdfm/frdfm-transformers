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


class FactorizedPoseTransformer(nn.Module):
    """Independent shared clause encoder with learned body-part routing and angle prediction."""
    def __init__(self, width=48, layers=1, heads=4):
        super().__init__()
        from text2image.stick_figures.data import CLAUSE_TEXT
        self.config = dict(width=width, layers=layers, heads=heads)
        self.embed = nn.Embedding(len(VOCAB) + 2, width, padding_idx=PAD)
        self.position = nn.Embedding(CLAUSE_TEXT, width)
        layer = nn.TransformerEncoderLayer(width, heads, width * 4, dropout=0.1,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.part_head = nn.Linear(width, 5)
        self.angle_head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

    def details(self, tokens):
        batch, clauses, length = tokens.shape
        flat = tokens.reshape(batch * clauses, length)
        h = self.embed(flat) + self.position.weight[:length]
        h = self.encoder(h, src_key_padding_mask=flat == PAD)
        encoded = self.norm(h[:, 0]).reshape(batch, clauses, -1)
        identity_logits = self.part_head(encoded)
        angles = self.angle_head(encoded).squeeze(-1)
        # Soft assignment to output slots; no rule maps a word to a body part or angle.
        routes = identity_logits.softmax(-1)
        routes = routes / routes.sum(dim=1, keepdim=True).clamp_min(1e-8)
        controls = (routes * angles.unsqueeze(-1)).sum(dim=1)
        return controls, identity_logits, angles

    def forward(self, tokens):
        return self.details(tokens)[0]


def model_class(architecture):
    if architecture == "pooled":
        return PoseTransformer
    if architecture == "factorized":
        return FactorizedPoseTransformer
    raise ValueError(f"Unknown architecture: {architecture}")
