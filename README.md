# An English → French Transformer, built from scratch

A complete EN→FR translator in ~200 lines of PyTorch — no `nn.Transformer`, no training framework. This README explains it top-down: one box at a time, each opening only boxes the previous step introduced.

```bash
python prepare_data.py   # opus-100 en-fr (~1M pairs), shared 16k BPE, data/*.bin
python train.py          # 5,000 steps, ~16.6M params
python translate.py "the weather is nice today"
```

Data facts used below: sentences are BPE-tokenized to at most **60 tokens**, `PAD = 0`, `EOS = 1`, vocab = **16,000** ([prepare_data.py](prepare_data.py)).

---

# Part 1 — the architecture, one box at a time

Shapes use one running example: English = 6 tokens + EOS, French = 5 tokens + EOS.

## Step 1 — The whole model is one box

```
              ┌─────────────────────────────────────┐
 EN ids ──────▶   "score the next French token"     ──────▶ FR logits
 (1, 7)                                                 (1, 5, 16000)
              └─────────────────────────────────────┘
```

The surprising first fact: **the model never "translates" anything in one shot.** Its only job is to do fill-in-the-blank: *given the English sentence and some French words so far, how likely is each of the 16,000 vocabulary tokens to be the next French word?* A translation is then produced *outside* the model, by `translate.py`: ask the box, take the most likely token, append it, ask again — until it says the "end" token.

Zoom constants for the rest of the tour:

| Constant | Value |
|---|---|
| `VOCAB_SIZE` | 16,000 |
| `D_MODEL` | 256 |
| `N_HEADS` | 8 (32 dims each) |
| `N_LAYERS` | 4 per stack |
| `FF_DIM` | 1024 |
| `MAX_LEN` | 60 |

## Step 2 — Open the box: a reading room, a writing room, one bridge

`Model.forward` ([train.py:106-113](train.py#L106-L113)) is literally three lines:

```python
src_pad_mask = src == 0
memory = self.encoder(src)
logits = self.decoder(tgt_in, memory, src_pad_mask)
```

Why two separate machines instead of one? Think about what each side is allowed to see:

- The **encoder** reads the *whole* English sentence at once — when you read an English rendition, every word is already laid out, left to right, no future to hide. Its output is `memory`: one 256-dim "note" per English position, `(1, 7, 256)`.
- The **decoder** writes French *causally* — a token must not see words that come after it (that's how translators actually behave mid-sentence).
- The **bridge**: the decoder never touches English directly. It only sees the encoder's notes. That's why `Model` takes `memory` as a separate argument — there is no shared data anywhere else.

## Step 3 — Words have to become numbers first

Both rooms start the same way: look each token up in a big table (16000 rows × 256 columns = the embedding), add a second table telling the model *which position* this word sits at, and you get a bundle of 256 numbers per word — the **residual stream**.

The intuitive way to picture what follows: the residual stream is a running note-pad of width 256, and every layer only *jots corrections onto it* (`h = h + something`). No layer ever replaces what's written. That's why 4 blocks can simply run in sequence and the final state is "the original words, enriched 8 times".

One sneak preview of the endings: the decoder's final "score every vocab word" layer turns out to be **the same table flipped sideways** — output scores = dot product of the running state against every word vector. Called *weight tying*; it means input and output share one dictionary, halving the biggest tables and forcing the dictionary rows to be *productions the model can actually emit*.

## Step 4 — Open a block: mix ideas, peek at the notes, then reflect

Each of the 4 blocks (in *each* room) is the same three-move routine:

```python
attn_out, _ = self.self_attn(h, h, h, attn_mask=attn_mask)
h = self.ln_1(h + attn_out)                                  # ① mix
if memory is not None:                                       # encoder skips this
    attn_out, _ = self.cross_attn(h, memory, memory, key_padding_mask=src_pad_mask)
    h = self.ln_2(h + attn_out)                              # ② peek at notes
return self.ln_3(h + self.ff_out(F.gelu(self.ff(h))))        # ③ reflect
```

The three moves:

- ① **Self-attention — the words talk to each other.** Every word poses a "question" about every other word and pulls in the answers, weighted by relevance. "It" gets to decide how much of "the weather" it should soak up. attention = 8 independent questions simultaneously (*who's the subject? what verb owns me? which noun am I describing?*), each using a 32-dim slice, then merged. The decoder's version hides the future: `torch.triu(..., diagonal=1)` — word *i* sees words `0..i` only.
- ② **Cross-attention — the French word checks its notes.** *Which English words am I here because of?* The French position queries English `memory`; the whole English sentence is visible because it's fully known. `key_padding_mask` hides PAD filler slots.
- ③ **Feed-forward — time to reflect alone.** Each position, independently, expands 256→1024 numbers, bends them (GELU), squeezes back to 256. No other positions are involved: mixing happened in ① and ②; ③ is quiet thinking.

After each move: add the move's result to the stream, then renormalize — LayerNorm sits *after* the add (**post-LN**, as the original 2017 paper; everything modern uses pre-LN). Intuition for why it exists: the word "add" means the stream volume can wander upward block after block; the norm clamps it back to a healthy amplitude.

One quirk: because one `Layer` class serves both rooms, the encoder carries cross-attention weights it never uses (~1M dead params, no gradient, AdamW skips them — harmless, just wasteful).

## Step 5 — How big is this thing?

**Total: 16,650,240 parameters (~16.6M),** built entirely from the tables above:

| Part | Params |
|---|---:|
| Token embed, 2 stacks × (16000×256) | 8,192,000 |
| Pos embed, 2 stacks × (60×256) | 30,720 |
| Layers, 8 × 1,053,440 (4 per stack) | 8,427,520 |
| **Total** | **16,650,240** |

| Inside one Layer (same for all 8) | Params |
|---|---:|
| self_attn: 3·(256×256) W + 768 b + out 256×256 W + 256 b | 263,168 |
| cross_attn: as above | 263,168 |
| ff: 256×1024 W + 1024 b | 263,168 |
| ff_out: 1024×256 W + 256 b | 262,400 |
| 3 × LayerNorm(256) | 1,536 |
| **per Layer** | **1,053,440** |

Roughly half the parameters are just the two dictionary tables — at this scale, "knowing words" costs more attention-lookup than the attention itself.

---

# Part 2 — Training: how it learned, in plain terms

The loop ([train.py:178-215](train.py#L178-L215)) has no framework: it's a textbook training loop you can read top to bottom.

| Constant | Value |
|---|---|
| `BATCH_SIZE` | 192 sampled at random, *with replacement* (no epochs) |
| `total_steps` | 5,000 |
| `LEARNING_RATE` | 5e-4 peak |
| `WARMUP_STEPS` | 100 |
| `GRAD_CLIP` | 1.0 global norm |
| optimizer | AdamW(β=(0.9, 0.95), eps=1e-8) |
| precision | bf16 forward on CUDA, fp32 loss + optimizer; TF32 on |

## The exam (the loss)

Every step: take 192 random English/French sentence pairs, and quiz the model — French grammar is hidden, French token *i* is shown to it, and it must guess token *i+1*:

```python
tgt_in  = tgt[:, :-1]                   # input: French tokens 0..T-2
target  = ids_to_matchable(tgt[:, 1:])  # the answers; PAD → -100 (don't grade)
loss    = F.cross_entropy(logits.float().reshape(-1, 16000), target.reshape(-1))
```

Three human-intuition notes:

- **Teacher forcing** — during training we *cheat* and hand the model the true French prefix ("…so far it should have written 'il fait'"), so all positions can be graded in one parallel pass, instead of all the wrong prefixes it would actually generate. The trade-off: train on the true history, deploy on its own history.
- **−100 grades are ignored** — `cross_entropy` treats `-100` as `ignore_index`, so PAD filler positions neither help nor hurt; `.float()` forces the score arithmetic slow-and-precise in fp32 while the matrix math above ran fast bf16.
- **EOS is part of the exam** — the model must learn to say `EOS = 1`, i.e. "my sentence is done" — because exactly that statement is what stops `translate.py`'s loop. A model that never learns EOS produces word salad forever.

Rows get trimmed to the longest real sentence in the batch; a 34-token batch does 34-wide compute, not 60.

## The walking pace (the LR schedule)

```python
lr = LEARNING_RATE * min(1.0, (step + 1) / WARMUP_STEPS)
```

Learning rate = how big each corrective step is. The model starts as an infant: take tiny steps (linear warmup, ~0 → 5e-4 over 100 steps) precisely because the post-LN stack from Part 1 is unstable before its norms settle in. After that: **a steady 5e-4, forever.** No decay — this is a fixed 5,000-step sprint, so it's a straight road, not a cosine hill.

The value is written by hand into `optimizer.param_groups[0]["lr"]` each step; AdamW uses whatever the current group LR is on `optimizer.step()`.

## Checkpointing and pop quizzes

Every **10 steps** (500 times total): a no-grades pass over all 4,000 unseen validation pairs — an honest "can you actually translate, not just memorize" check — appended to `logs/train.log` along with the current checkpoint, saved *over* `checkpoints/model.pt` (one latest-only file, no keeps-best bookkeeping).

## The whole thing on one screen

```python
for step in range(5_000):
    optimizer.param_groups[0]["lr"] = 5e-4 * min(1.0, (step + 1) / 100)
    src, tgt = get_batch(...)          # 192 random pairs
    src, tgt = trim_batch(src, tgt)
    tgt_in = tgt[:, :-1]               # cheat: we know the French so far
    with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=USE_BF16):
        logits = model(src, tgt_in)
    loss = F.cross_entropy(logits.float().reshape(-1, 16000), ids_to_matchable(tgt[:, 1:]).reshape(-1))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()                    # outside autocast, per PyTorch convention
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # don't overcorrect
    optimizer.step()
    if (step + 1) % 10 == 0: eval, log, save checkpoint
```

---

# Inference

[translate.py](translate.py) encodes the source once, seeds the target with `[EOS]` (stand-in BOS), and decodes **greedily**: always take the single most likely next token, append it, re-ask — an innocent strategy that works because training made `EOS` a real, learnable target (Part 2's exam explicitly grades it).
