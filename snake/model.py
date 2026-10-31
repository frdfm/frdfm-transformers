"""Matrix history encoder and autoregressive ten-action Transformer decoder."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .game import Config
from .planning import PlanGeometry

ROOT = Path(__file__).resolve().parent
HORIZON = 10
BOS = 3
CHECKPOINT = ROOT / 'checkpoints' / 'seq10-wide-v2.pt'


class SnakeTransformer(nn.Module):
    def __init__(self, height=20, width=20, history=2, dim=64, layers=2, patch=4, horizon=HORIZON):
        super().__init__()
        if min(height, width, history, dim, layers, patch) < 1 or height % patch or width % patch or dim % 4 or horizon != HORIZON:
            raise ValueError('Board dimensions must divide into patches; dim must be divisible by 4')
        self.config = dict(height=height, width=width, history=history,
                           dim=dim, layers=layers, patch=patch, horizon=horizon)
        self.project = nn.Linear(8 * patch * patch, dim)
        self.space = nn.Parameter(torch.randn(1, 1, height // patch * (width // patch), dim) * 0.02)
        self.time = nn.Parameter(torch.randn(1, history, 1, dim) * 0.02)
        block = nn.TransformerEncoderLayer(dim, 4, dim * 3, dropout=0.0,
                                            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.moves = nn.Embedding(4, dim)
        self.positions = nn.Embedding(horizon, dim)
        decoder = nn.TransformerDecoderLayer(dim, 4, dim * 3, dropout=0.0,
                                             batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(decoder, layers)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 3))

    def encode(self, matrices):
        c = self.config
        b, t, h, w = matrices.shape
        if (t, h, w) != (c['history'], c['height'], c['width']):
            raise ValueError('Expected [batch, history, height, width] matching model config')
        p = c['patch']
        x = F.one_hot(matrices.long(), 8).float()
        x = x.reshape(b, t, h // p, p, w // p, p, 8)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).reshape(b, t, -1, p * p * 8)
        x = (self.project(x) + self.space + self.time).flatten(1, 2)
        return self.encoder(x)

    def decode(self, tokens, memory):
        length = tokens.shape[1]
        if not 1 <= length <= self.config['horizon']:
            raise ValueError('Decoder prefix must fit the action horizon')
        x = self.moves(tokens) + self.positions(torch.arange(length, device=tokens.device))
        mask = torch.triu(torch.ones(length, length, device=tokens.device, dtype=torch.bool), diagonal=1)
        return self.head(self.decoder(x, memory, tgt_mask=mask))

    def forward(self, matrices, tokens):
        return self.decode(tokens, self.encode(matrices))

    @torch.inference_mode()
    def generate(self, matrices, constrain=True):
        memory = self.encode(matrices)
        geometry = [PlanGeometry(grid) for grid in matrices[:, -1].cpu().numpy()] if constrain else None
        tokens = torch.full((len(matrices), 1), BOS, dtype=torch.long, device=matrices.device)
        probabilities = []
        for tick in range(self.config['horizon']):
            logits = self.decode(tokens, memory)[:, -1]
            if geometry is not None:
                allowed = torch.tensor([g.allowed(HORIZON - tick) for g in geometry],
                                       device=logits.device, dtype=torch.bool)
                logits = logits.masked_fill(~allowed, float('-inf'))
            probs = logits.softmax(-1)
            probabilities.append(probs)
            moves = probs.argmax(-1, keepdim=True)
            tokens = torch.cat((tokens, moves), dim=1)
            if geometry is not None:
                for g, move in zip(geometry, moves.flatten().tolist()):
                    g.advance(move)
        return tokens[:, 1:], torch.stack(probabilities, dim=1)


def load(path=CHECKPOINT, device='cpu'):
    state = torch.load(path, map_location=device, weights_only=True)
    if state['config'].get('horizon') != HORIZON:
        raise ValueError('Expected a ten-action checkpoint. Run python -m snake.train first.')
    if state.get('experiment') != 'snake-seq10':
        raise ValueError('Expected a Snake ten-action checkpoint')
    game_config = Config(**state['game_config'])
    if any(state['config'][key] != getattr(game_config, key) for key in ('height', 'width', 'history')):
        raise ValueError('Model and game configurations differ')
    model = SnakeTransformer(**state['config']).to(device)
    model.load_state_dict(state['model'])
    return model.eval(), game_config


class ModelPlayer:
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        self.probabilities = [0.0] * 3
        self.refills = 0
        self.reset()

    def reset(self):
        self.queue = []
        self.position = 0
        self.queue_probabilities = []
        self.probabilities = [0.0] * 3

    @property
    def needs_plan(self):
        return self.position == len(self.queue)

    @torch.inference_mode()
    def __call__(self, observation=None):
        if self.needs_plan:
            if observation is None:
                raise ValueError('A fresh matrix history is required to refill the queue')
            x = torch.as_tensor(observation, device=self.device).unsqueeze(0)
            moves, probs = self.model.generate(x)
            self.queue = moves[0].tolist()
            self.queue_probabilities = probs[0].tolist()
            self.position = 0
            self.refills += 1
        move = self.queue[self.position]
        self.probabilities = self.queue_probabilities[self.position]
        self.position += 1
        return move
