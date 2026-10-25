"""Read-only, CPU/FP32 worked batch for README; no checkpoint is loaded or saved."""
import json
from pathlib import Path
import math

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from en2fr.model import Model
from en2fr.pipeline import PAD, BOS
from en2fr.train import trim_batch, ids_to_matchable


def expanded_attention(module, query, key_value, padding=None, causal=None):
    width, heads = module.embed_dim, module.num_heads
    weights, biases = module.in_proj_weight.chunk(3), module.in_proj_bias.chunk(3)
    projected = [F.linear(x, w, b) for x, w, b in zip((query, key_value, key_value), weights, biases)]
    q, k, v = [x.reshape(x.shape[0], x.shape[1], heads, width // heads).transpose(1, 2) for x in projected]
    scores = q @ k.transpose(-2, -1) / math.sqrt(width // heads)
    masked = scores.clone()
    if padding is not None:
        masked = masked.masked_fill(padding[:, None, None, :], -torch.inf)
    if causal is not None:
        masked = masked.masked_fill(causal[None, None], -torch.inf)
    probabilities = masked.softmax(-1)
    combined = (probabilities @ v).transpose(1, 2).reshape(query.shape[0], query.shape[1], width)
    output = F.linear(combined, module.out_proj.weight, module.out_proj.bias)
    actual = module(query, key_value, key_value, key_padding_mask=padding,
                    attn_mask=causal, need_weights=False)[0]
    torch.testing.assert_close(output, actual, atol=2e-6, rtol=2e-5)
    return scores, masked, probabilities, v


def main():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    folder = Path(__file__).resolve().parent / "data"
    tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
    row_ids = [2, 15]
    arrays = [np.memmap(folder / f"train_{side}.bin", dtype=np.uint16, mode="r").reshape(-1, 60)[row_ids].copy()
              for side in ("src", "tgt")]
    source, french = trim_batch(*(torch.tensor(a.astype(np.int64)) for a in arrays))
    decoder_input = torch.cat([torch.full_like(french[:, :1], BOS), french[:, :-1]], 1)
    expected = ids_to_matchable(french)
    padding = source == PAD
    model = Model().eval()
    shapes = {}
    handles = []
    for stack in ("encoder", "decoder"):
        for index, layer in enumerate(getattr(model, stack).layers):
            name = f"{stack}_block_{index + 1}"
            handles.append(layer.register_forward_hook(lambda m, i, o, name=name: shapes.update({name: list(o.shape)})))
    logits = model(source, decoder_input)
    for handle in handles:
        handle.remove()
    logits.retain_grad()
    losses = F.cross_entropy(logits.reshape(-1, 16000), expected.reshape(-1),
                             ignore_index=-100, reduction="none").reshape_as(expected)
    valid = expected != -100
    loss = F.cross_entropy(logits.reshape(-1, 16000), expected.reshape(-1), ignore_index=-100)
    torch.testing.assert_close(loss, losses.sum() / valid.sum())
    loss.backward()
    assert torch.all(logits.grad[~valid] == 0)
    with torch.no_grad():
        h = model.encoder.embed(source) + model.encoder.pos.weight[:source.shape[1]]
        enc = expanded_attention(model.encoder.layers[0].self_attn, h, h, padding)
        memory = model.encoder(source)
        h = model.decoder.embed(decoder_input) + model.decoder.pos.weight[:decoder_input.shape[1]]
        length = decoder_input.shape[1]
        causal = torch.ones(length, length, dtype=torch.bool).triu(1)
        self_out = model.decoder.layers[0].self_attn(h, h, h, attn_mask=causal, need_weights=False)[0]
        after_self = model.decoder.layers[0].ln_1(h + self_out)
        cross = expanded_attention(model.decoder.layers[0].cross_attn, after_self, memory, padding)
    def attention_row(result, position):
        return {"scores": result[0][0, 0, position].tolist(),
                "masked_scores": [round(x, 6) if math.isfinite(x) else "-inf" for x in result[1][0, 0, position].tolist()],
                "weights": result[2][0, 0, position].tolist()}
    print(json.dumps({
        "seed": 7, "checkpoint": "none: randomly initialized model, CPU float32",
        "dataset_rows_zero_based": row_ids,
        "sentences": [{"english": tokenizer.decode(a.tolist()), "french": tokenizer.decode(b.tolist())} for a, b in zip(source, french)],
        "source": source.tolist(), "french_tokens": french.tolist(),
        "decoder_input": decoder_input.tolist(), "expected_tokens": expected.tolist(),
        "pieces": [{"english": [tokenizer.id_to_token(i) for i in a.tolist()],
                    "french": [tokenizer.id_to_token(i) for i in b.tolist()]} for a, b in zip(source, french)],
        "source_padding_mask": padding.tolist(), "block_shapes": shapes,
        "encoder_block1_head0_query1": attention_row(enc, 1),
        "decoder_block1_cross_head0_query0": attention_row(cross, 0),
        "loss_per_position": losses.tolist(), "valid_positions": int(valid.sum()),
        "loss_sum": float(losses.detach().sum()), "mean_loss": float(loss.detach()),
        "ignored_logits_gradient_max": float(logits.grad[~valid].abs().max()),
        "first_logit_row_top5_ids": logits[0, 0].topk(5).indices.tolist(),
    }, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
