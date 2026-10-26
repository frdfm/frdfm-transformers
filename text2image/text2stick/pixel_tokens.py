"""Lossless 2x2 binary-pixel packing. No learned codebook or body geometry."""
import torch
from text2image.text2stick.tokenization import SIDE,PATCH,PIXELS,IMAGE_TOKENS


def pack_pixels(pixels):
    grid=pixels.reshape(-1,SIDE//PATCH,PATCH,SIDE//PATCH,PATCH)
    bits=grid.permute(0,1,3,2,4).reshape(-1,IMAGE_TOKENS,PATCH*PATCH).long()
    weights=2**torch.arange(PATCH*PATCH,device=pixels.device)
    return (bits*weights).sum(-1)


def unpack_tokens(tokens):
    bits=(tokens.long().unsqueeze(-1)>>torch.arange(PATCH*PATCH,device=tokens.device))&1
    return bits.reshape(-1,SIDE//PATCH,SIDE//PATCH,PATCH,PATCH).permute(0,1,3,2,4).reshape(-1,PIXELS)
