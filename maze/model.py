import torch
from torch import nn


class MazeTransformer(nn.Module):
    def __init__(self, size=7, width=96, layers=2):
        super().__init__()
        self.config = dict(size=size, width=width, layers=layers)
        self.cells = nn.Embedding(4, width)
        self.rows = nn.Embedding(size, width)
        self.cols = nn.Embedding(size, width)
        self.moves = nn.Embedding(7, width)
        self.positions = nn.Embedding(size*size+2, width)
        self.transformer = nn.Transformer(d_model=width, nhead=4, num_encoder_layers=layers,
                                          num_decoder_layers=layers, dim_feedforward=width*4,
                                          dropout=0.0, batch_first=True)
        self.head = nn.Linear(width, 5)

    def encode(self, grids):
        size = self.config['size']
        index = torch.arange(size*size, device=grids.device)
        x = self.cells(grids.flatten(1)) + self.rows(index // size) + self.cols(index % size)
        return self.transformer.encoder(x)

    def decode(self, tokens, memory):
        n = tokens.shape[1]
        x = self.moves(tokens) + self.positions(torch.arange(n, device=tokens.device))
        mask = torch.triu(torch.ones(n, n, device=tokens.device, dtype=torch.bool), diagonal=1)
        return self.head(self.transformer.decoder(x, memory, tgt_mask=mask))

    def forward(self, grids, tokens):
        return self.decode(tokens, self.encode(grids))


def load(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = MazeTransformer(**checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['model'])
    return model.eval()
