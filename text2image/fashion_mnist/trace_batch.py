"""Read-only two-image trace for README. CPU FP32, random weights, no saved-state changes."""
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from text2image.fashion_mnist.model import ImageTransformer, encode_prompt, CLASSES, PAD, PREFIX
from text2image.fashion_mnist.train import pixel_loss


def attention_probe(module, h, tokens, query_positions):
    batch, length, width = h.shape
    heads = module.num_heads
    weights, biases = module.in_proj_weight.chunk(3), module.in_proj_bias.chunk(3)
    q, k, v = [F.linear(h, w, b).reshape(batch, length, heads, width // heads).transpose(1, 2)
               for w, b in zip(weights, biases)]
    q = q[:, :, query_positions]
    raw = q @ k.transpose(-2, -1) / math.sqrt(width // heads)
    causal = torch.arange(length)[None, :] > torch.tensor(query_positions)[:, None]
    blocked = (tokens == PAD)[:, None, None, :] | causal[None, None]
    masked = raw.masked_fill(blocked, -torch.inf)
    probabilities = masked.softmax(-1)
    combined = (probabilities @ v).transpose(1, 2).reshape(batch, len(query_positions), width)
    expanded = F.linear(combined, module.out_proj.weight, module.out_proj.bias)
    actual = module(h[:, query_positions], h, h, key_padding_mask=tokens == PAD,
                    attn_mask=causal, need_weights=False)[0]
    torch.testing.assert_close(expanded, actual, atol=2e-6, rtol=2e-5)
    assert torch.all(probabilities.masked_select(blocked.expand_as(probabilities)) == 0)
    return {f"example_{b}_head_0_query_{position}": {
        "raw_scores_first_12": raw[b, 0, j, :12].tolist(),
        "weights_first_12": probabilities[b, 0, j, :12].tolist(),
        "allowed_key_count": int((~blocked[b, 0, j]).sum()),
        "masked_weight_max": float(probabilities[b, 0, j].masked_select(blocked[b, 0, j]).max()),
    } for b in range(batch) for j, position in enumerate(query_positions)}


def main():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    path = Path(__file__).resolve().parent / "data" / "28" / "train.npz"
    with np.load(path) as data:
        indices = [int(np.flatnonzero(data["labels"] == label)[0]) for label in (8, 4)]
        pixels = torch.tensor(data["pixels"][indices].astype(np.int64))
    prompts = ["draw a bag", "a grayscale image of a coat"]
    captions = torch.tensor([encode_prompt(text) for text in prompts])
    tokens = torch.cat([captions, pixels[:, :-1]], dim=1)
    model = ImageTransformer(side=28).eval()
    shapes, handles = {}, []
    for i, block in enumerate(model.blocks):
        handles.append(block.register_forward_hook(lambda m, args, out, i=i: shapes.update({f"block_{i+1}": list(out.shape)})))
    logits = model(tokens)
    for handle in handles:
        handle.remove()
    logits.retain_grad()
    selected = logits[:, PREFIX - 1:]
    losses = F.cross_entropy(selected.reshape(-1, 16), pixels.reshape(-1), reduction="none").reshape_as(pixels)
    loss = losses.mean()
    torch.testing.assert_close(loss, pixel_loss(model, captions, pixels))
    loss.backward()
    assert torch.all(logits.grad[:, :PREFIX - 1] == 0)
    with torch.no_grad():
        embedded = model.embed(tokens) + model.position.weight[:tokens.shape[1]]
        normalized = model.blocks[0].norm1(embedded)
        probes = attention_probe(model.blocks[0].attention, normalized, tokens, [9, 10])
    image_info = []
    for b in range(2):
        nonzero = torch.nonzero(pixels[b], as_tuple=False).flatten()
        first = int(nonzero[0])
        image_info.append({"dataset_row": indices[b], "category": CLASSES[(8,4)[b]],
                           "first_28_pixels": pixels[b, :28].tolist(),
                           "first_nonzero_index": first, "first_nonzero_row_column": [first // 28, first % 28],
                           "window_start": max(0, first - 3),
                           "first_nonzero_window": pixels[b, max(0,first-3):first+9].tolist(),
                           "last_pixel": int(pixels[b, -1]),
                           "zero_pixels": int((pixels[b] == 0).sum()),
                           "loss_sum": float(losses[b].detach().sum()),
                           "mean_loss": float(losses[b].detach().mean()),
                           "first_6_losses": losses[b, :6].detach().tolist(),
                           "loss_at_first_nonzero": float(losses[b, first].detach())})
    print(json.dumps({"seed": 7, "checkpoint": "none; random model in eval mode, CPU FP32",
                      "images": image_info, "prompts": prompts, "captions": captions.tolist(),
                      "input_shape": list(tokens.shape), "caption_padding_mask": (captions==PAD).tolist(),
                      "block_shapes": shapes, "all_logits_shape": list(logits.shape),
                      "selected_logits_shape": list(selected.shape), "attention": probes,
                      "loss_sum": float(losses.detach().sum()), "pixel_count": pixels.numel(),
                      "mean_loss": float(loss.detach()),
                      "excluded_prefix_logit_gradient_max": float(logits.grad[:, :9].abs().max()),
                      "last_caption_logit_gradient_norm": float(logits.grad[:, 9].norm()),
                      "pad_embedding_gradient_norm": float(model.embed.weight.grad[PAD].norm())}, indent=2))


if __name__ == "__main__":
    main()
