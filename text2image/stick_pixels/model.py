"""Direct autoregressive pixel model: text + earlier pixels -> next binary pixel."""
import torch
from torch import nn
from text2image.stick_pixels.schema import INPUT_VOCAB, PAD, PREFIX, PIXELS


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(width, width*4), nn.GELU(), nn.Linear(width*4,width))

    def forward(self, h, causal, padding):
        x = self.norm1(h)
        h = h + self.attention(x,x,x,attn_mask=causal,key_padding_mask=padding,need_weights=False)[0]
        return h + self.ff(self.norm2(h))


class PixelTransformer(nn.Module):
    def __init__(self, width=64, layers=3, heads=4):
        super().__init__()
        self.config = dict(width=width,layers=layers,heads=heads)
        self.embed = nn.Embedding(INPUT_VOCAB,width)
        self.position = nn.Embedding(PREFIX+PIXELS,width)
        self.blocks = nn.ModuleList([Block(width,heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width,2)

    def forward(self,tokens):
        length=tokens.shape[1]
        h=self.embed(tokens)+self.position.weight[:length]
        causal=torch.ones(length,length,dtype=torch.bool,device=tokens.device).triu(1)
        for block in self.blocks:
            h=block(h,causal,tokens==PAD)
        return self.output(self.norm(h))

    @torch.inference_mode()
    def sample(self,captions,temperature=0.8):
        if temperature < 0:
            raise ValueError("Temperature must be nonnegative; zero selects greedy pixels")
        tokens=captions
        for _ in range(PIXELS):
            logits=self(tokens)[:,-1]
            pixel=logits.argmax(-1,keepdim=True) if temperature==0 else torch.multinomial((logits/temperature).softmax(-1),1)
            tokens=torch.cat([tokens,pixel],1)
        return tokens[:,PREFIX:]
