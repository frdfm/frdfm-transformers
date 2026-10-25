# English -> French: an encoder-decoder Transformer

A small translator trained locally from scratch on paired examples: an English sentence and its French translation. A Transformer is a neural network that learns which parts of a sequence matter when interpreting or producing another part. This guide first explains the translation process, then opens the components used to implement it in PyTorch.

The model has **16,650,240 parameters**. It is an educational translator intended to produce some useful short translations, not a replacement for a large pretrained translation system.

## Start with the words used in this guide

The computer does not receive words directly. A **tokenizer** splits text into pieces called **tokens**, then gives each piece an integer identifier, or **ID**. A token may be a word, part of a word, punctuation, or bytes representing a character. A **vocabulary** is the set of pieces and special markers the tokenizer knows. The network learns a vector of numbers for each ID; this vector is its **embedding**.

**BPE** means **byte-pair encoding**: tokenizer training repeatedly merges frequent neighboring pieces into larger pieces. Here it starts from bytes, the units that represent text in computer memory. The same tokenizer handles English and French, so we call it *shared*. Sharing the tokenizer does not mean sharing all of the model's weights.

There are also special markers that are not ordinary words:

| Name used in code | Full meaning | How to read it in an example |
|---|---|---|
| `BOS` | Beginning of sequence | "Start writing French now" |
| `EOS` | End of sequence | "This sentence has ended" |
| `PAD` | Padding | "This slot is unused; ignore it where appropriate" |
| `UNK` | Unknown token | "A piece the tokenizer could not represent" |

**Encoding text into token IDs** is the tokenizer's job. The **encoder network** is a different component: it transforms those IDs into contextual representations of the English sentence. The **decoder network** uses these representations to predict French. Finally, the tokenizer's **text decoder** turns generated IDs back into readable text. These three uses of "encode/decode" are related but distinct.

A **parameter**, also called a weight when referring to learned connection values, is an adjustable number inside the network. "Trained from scratch" means these numbers start without pretrained language knowledge and are adjusted using this project's examples.

## Run it

From the repository root with your venv active:

```powershell
python -m en2fr.prepare_data
python -m en2fr.train --steps 10000
python -m en2fr.translate "the weather is nice today"
```

Dependencies: `torch`, `numpy`, `datasets`, and `tokenizers`. Preparation requires internet access. Existing compatible data and checkpoints can be reused without rebuilding them.

All artifacts are relative to this folder, regardless of the shell's working directory:

| Path | Contents |
|---|---|
| `data/tokenizer.json` | Shared English/French byte-level BPE tokenizer |
| `data/metadata.json` | Data format, dimensions, tokenizer hash |
| `data/train_src.bin`, `data/train_tgt.bin` | Padded training token rows |
| `data/val_src.bin`, `data/val_tgt.bin` | Padded validation token rows |
| `checkpoints/model.pt` | Latest saved training state |
| `logs/train.log` | Step, training loss, validation loss, elapsed seconds |

Preparation replaces data and the tokenizer. A fresh training run replaces the log and checkpoint; `--resume` preserves the training trajectory and appends to the log.

## Worked forward pass: two real sentence pairs, all the way to loss

This section follows **one batch of two examples** through the current code, in execution order. It shows the integer inputs, all four masking/ignore applications, encoder and decoder calculations, and the scalar loss. The training default is 192 examples; using two here changes only the batch dimension and makes the details readable.

### Reproduce this walkthrough

```powershell
python -m en2fr.trace_batch
```

[trace_batch.py](trace_batch.py) reads rows **2 and 15, zero-based**, from the local prepared training files. It uses their real text and tokenizer IDs, a randomly initialized model with PyTorch seed 7, and CPU 32-bit floating point. **It does not load, modify, or save your trained checkpoint.** It verifies an expanded attention calculation against PyTorch, runs the forward pass, calculates loss, and checks gradients at ignored outputs. The backward calculation is only a check; no optimizer update is performed.

The numerical outputs below were measured using the local tokenizer/data and this trace. Token IDs can change after tokenizer training; floating-point results can vary slightly with software/hardware. This is a correctness demonstration, not a translation-quality example. A trained model has different scores, attention weights, and loss. Normal GPU training also uses BF16 for selected operations; this trace uses FP32 throughout to make inspection easier.

### 1. The batch begins as two translation pairs

| Batch row | English source | French translation |
|---|---|---|
| 0 | `Come on.` | `Venez !` |
| 1 | `This is my office.` | `C'est mon bureau.` |

A **tokenizer** splits text into pieces and assigns an integer **token identifier (ID)** to each. This tokenizer uses **byte-pair encoding (BPE)**, which learns frequent pieces by merging neighboring byte sequences. The same tokenizer handles both languages, but the model has separate English and French embedding tables.

These are the measured pieces and IDs. `␠` below means a literal leading space in the piece; it is explanatory notation, not a separate token:

| Row and language | Token pieces before adding special markers | IDs |
|---|---|---|
| 0 English | `Come`, `␠on`, `.` | `[3874, 363, 17]` |
| 0 French | `V`, `enez`, `␠!` | `[57, 3926, 779]` |
| 1 English | `This`, `␠is`, `␠my`, `␠office`, `.` | `[1080, 400, 895, 4194, 17]` |
| 1 French | `C`, `'`, `est`, `␠mon`, `␠bureau`, `.` | `[38, 10, 344, 927, 4580, 17]` |

Notice that `Venez` takes **two tokens**, and `C'est` takes **three**. Sequence lengths count tokens, not words or characters. An ID is an address in a learned lookup table, not a numerical measurement of a word's meaning.

Special IDs are `PAD = 0` (unused slot), `EOS = 1` (end of sequence), and `BOS = 2` (beginning of French decoding). Ordinary English and French sentences both receive EOS. French BOS is inserted later when constructing the decoder input, not stored in the target file.

### 2. Stored padding and batch trimming

Preparation appends EOS and right-pads each source and target row to 60 IDs:

```text
stored English, shape [2,60]:
row 0: [3874, 363, 17, 1, 0, 0, ...]
row 1: [1080, 400, 895, 4194, 17, 1, 0, ...]

stored French, shape [2,60]:
row 0: [57, 3926, 779, 1, 0, 0, ...]
row 1: [38, 10, 344, 927, 4580, 17, 1, 0, ...]
```

Padding makes batches rectangular arrays. It does not mean all sentences have the same real length. Files store unsigned 16-bit integers; `load_pairs` converts them to 64-bit integer tensors for embedding lookup. A **tensor shape** lists dimension sizes: `[2,6]` means two rows, six positions each.

The real source lengths are 4 and 6 including EOS; real French lengths are 4 and 7. `trim_batch` removes columns not needed by this batch. The current implementation keeps `max_source_length` source positions and `max_target_length + 1` French positions when possible, so this batch becomes:

```python
src = torch.tensor([
    [3874, 363,  17,    1,  0, 0],
    [1080, 400, 895, 4194, 17, 1],
])  # shape [2,6]

tgt = torch.tensor([
    [57, 3926, 779,   1,    0,  0, 0, 0],
    [38,   10, 344, 927, 4580, 17, 1, 0],
])  # shape [2,8]
```

`src` abbreviates source, English. `tgt` abbreviates target, the stored French data. The extra final French padding column is an implementation choice, not a mathematical requirement. Trimming changes batch tensor width; **masking later does not remove rows or columns**.

### 3. Construct what the decoder will receive

Before `Model.forward`, training shifts the French sequence right and prepends BOS:

```python
tgt_in = torch.cat([torch.full_like(tgt[:, :1], BOS), tgt[:, :-1]], dim=1)
```

`_in` means input: this is the target-language input to the decoder. With clearer descriptive aliases:

```text
french_tokens = tgt                            [2,8]
decoder_input = tgt_in                         [2,8]

row 0 input: [2, 57, 3926, 779,   1,    0,  0, 0]
row 1 input: [2, 38,   10, 344, 927, 4580, 17, 1]
```

The first four predictions in row 0 should be `V`, `enez`, `␠!`, EOS. We have not converted PAD to `-100` in any model input. That value is only introduced in the expected-answer copy for loss, after the forward pass in the actual loop.

### 4. Enter `Model.forward`: two inputs, one English padding mask

The actual call is:

```python
logits = model(src, tgt_in)
```

The model does **not** receive `tgt` or the expected answers as an additional argument. Inside [model.py](model.py):

```python
src_pad_mask = src == 0
memory = self.encoder(src)
logits = self.decoder(tgt_in, memory, src_pad_mask)
return logits
```

The source padding mask is a boolean tensor with shape `[2,6]`:

```text
row 0: [False, False, False, False, True, True]
row 1: [False, False, False, False, False, False]
```

For PyTorch `MultiheadAttention`, **True means blocked**. The mask is per example: row 0 has two English padding positions, row 1 has none. Positions in one example never attend to a different example in the batch.

The encoder recomputes the same `src == PAD` mask inside its own loop; the model-level copy above is passed to decoder cross-attention later. These equal masks have two separate uses. This is not a mask that hides entire padded sentences.

### 5. Encoder embedding lookup: `[2,6]` IDs become `[2,6,256]` vectors

The encoder token embedding table is `[16000,256]`: one learned vector of 256 numbers per vocabulary ID. The encoder position table is `[60,256]`: one vector per position.

```python
length = src.shape[1]  # 6
h = self.embed(src) + self.pos.weight[:length]
```

| Intermediate | Shape | Example |
|---|---|---|
| `src` | `[2,6]` | `src[0,1] = 363`, token `␠on` |
| `self.embed(src)` | `[2,6,256]` | Looks up row 363 for that position |
| `self.pos.weight[:6]` | `[6,256]` | Uses position rows 0-5 for both examples |
| Sum `h` | `[2,6,256]` | `h[0,1,:] = token_table[363,:] + position_table[1,:]` |

Broadcasting repeats the position table logically across the batch. These are two independent examples sharing the model parameters, not shared sentence contents.

The PAD slots also receive vectors. `padding_idx=0` does not remove them or implement attention masking. In this code, the explicit normal initialization overwrites the embedding's initial zero PAD row; position embeddings also make PAD representations nonzero. Correctness must come from the explicit masks, not an assumption that padding vectors are zero.

### 6. Encoder block 1: project queries, keys, and values

One **encoder block** contains multi-head self-attention, residual additions, layer normalizations, and a feed-forward network. There are four independent such blocks in sequence; every block processes both complete source rows.

For self-attention, all three projected inputs come from the same `h`:

```text
Q = linear_query(h)       [2,6,256]
K = linear_key(h)         [2,6,256]
V = linear_value(h)       [2,6,256]
```

A **query** expresses what a position seeks, a **key** is used to match it, and a **value** carries the information combined after matching. Each projection has its own learned weights and bias. PyTorch packs the three weight matrices into `in_proj_weight` of shape `[768,256]` and three biases into `[768]`; each weight slice is `[256,256]`.

For each projection, `linear(x) = x @ weight.T + bias`. The vectors are rearranged into eight heads of 32 numbers each:

```text
Q, K, V: conceptual head layout [2,8,6,32]
scores = Q @ K.transpose(-2,-1) / sqrt(32)
scores shape:                  [2,8,6,6]
```

The four dimensions are batch example, attention head, query position, and key position. A score row compares **one query against all six keys**. No dot product crosses the batch dimension. Dividing by `sqrt(32)` scales the scores; it does not normalize them into probabilities.

### 7. Mask application 1: block English PAD keys before softmax

The encoder calls self-attention with `key_padding_mask=src == PAD`. Expanded mathematical code, using the head layout above, is:

```python
scores = Q @ K.transpose(-2, -1) / math.sqrt(32)
scores = scores.masked_fill(src_pad_mask[:, None, None, :], float("-inf"))
weights = torch.softmax(scores, dim=-1)
head_output = weights @ V
```

`[2,6]` becomes `[2,1,1,6]` and broadcasts over eight heads and all six query rows. It blocks **columns** 4 and 5 for every query in example 0. Example 1 has no blocked source columns. Encoder attention has no causal mask: all nonpadding English positions may read each other.

Here is a measured row from encoder block 1, head 0, example 0, query position 1 (`␠on`), rounded:

| English key position | `Come` | `␠on` | `.` | EOS | PAD | PAD |
|---|---:|---:|---:|---:|---:|---:|
| Score before masking | -0.000411 | 0.000245 | 0.000380 | -0.000068 | -0.000262 | 0.000013 |
| Score after masking | -0.000411 | 0.000245 | 0.000380 | -0.000068 | negative infinity | negative infinity |
| Softmax weight | 0.249888 | 0.250052 | 0.250086 | 0.249974 | **0** | **0** |

Softmax is `exp(score) / sum(exp(scores))`; `exp(-infinity) = 0`. Therefore the corresponding value calculation is:

```text
head_output = 0.249888 * V_Come
            + 0.250052 * V_on
            + 0.250086 * V_period
            + 0.249974 * V_EOS
            + 0        * V_PAD_4
            + 0        * V_PAD_5
```

Each `V_*` is a 32-number vector for this head. Padded values still exist but contribute **exactly zero**. Their finite pre-mask dot products do not affect even the softmax denominator after replacement with negative infinity. They are not merely assigned small weights.

PyTorch performs this masking internally. Your code delegates it through:

```python
self.self_attn(
    h, h, h,
    attn_mask=None,
    key_padding_mask=src_pad_mask,
    need_weights=False,
)
```

The boolean mask is converted to an additive mask: zero for allowed keys, negative infinity for blocked ones. The optimized implementation may fuse operations, but it implements the same result. `need_weights=False` means do not return attention weights; it does not disable attention or masking. The trace script expands this calculation and checks it against the actual module output.

### 8. Finish encoder block 1, then run blocks 2-4

The weighted values are `[2,8,6,32]`. Merge the heads into `[2,6,256]`, then apply the learned attention output projection `[256,256]` with bias `[256]`. Next:

```text
attention output                         [2,6,256]
h1 = ln_1(original_h + attention_output)  [2,6,256]
f  = Linear(256 -> 1024)(h1)              [2,6,1024]
f  = GELU(f)                             [2,6,1024]
f  = Linear(1024 -> 256)(f)               [2,6,256]
block_output = ln_3(h1 + f)               [2,6,256]
```

A **residual addition** adds a correction without discarding the incoming representation. **LayerNorm**, layer normalization, normalizes each individual token vector, not the sentence or batch. For vector components `x_j` it computes:

```text
mean = mean(x over its 256 components)
variance = mean((x - mean)^2 over those components)
normalized_j = gamma_j * (x_j - mean) / sqrt(variance + 1e-5) + beta_j
```

`gamma` and `beta` are learned 256-element vectors. **GELU**, the Gaussian error linear unit, supplies nonlinearity; the default exact form is `0.5*x*(1 + erf(x/sqrt(2)))`. The feed-forward network transforms each position independently; attention is the part that mixes information across positions.

Encoder outputs measured by hooks:

| Stage | Shape |
|---|---|
| Encoder block 1 | `[2,6,256]` |
| Encoder block 2 | `[2,6,256]` |
| Encoder block 3 | `[2,6,256]` |
| Encoder block 4, returned as `memory` | `[2,6,256]` |

Block 2 receives block 1's vectors, not the original token IDs. All four have separate learned parameters. The English sequence is not split into four pieces, and eight heads do not mean eight blocks. No roles such as grammar or syntax are assigned to particular blocks in advance.

Our shared block class allocates cross-attention and `ln_2` even in the encoder, but skips them because `memory is None`. They receive no gradients and are unused parameter overhead. Each executed sublayer normalizes *after* residual addition: **post-LayerNorm**.

### 9. Decoder embedding lookup: French input is a different length

The decoder receives three things:

```text
tgt_in          [2,8]      French input IDs
memory          [2,6,256]  English encoder output
src_pad_mask    [2,6]      the original English padding pattern
```

It has its own `[16000,256]` token embedding and `[60,256]` position embedding. The same lookup-and-add calculation now produces `[2,8,256]`.

For example 0, input positions are `BOS, V, enez, ␠!, EOS, PAD, PAD, PAD`. The PAD embeddings are real vectors at this point. We do not remove them, and there is no `-100` input embedding. The loss will determine which predictions are graded later.

### 10. Mask application 2: causal French self-attention

Before decoder blocks run, the decoder constructs:

```python
causal = torch.triu(torch.ones(8, 8, dtype=torch.bool), diagonal=1)
```

Here `X` is boolean True, blocked; `.` is False, allowed. Rows are query positions; columns are key positions:

```text
query/key   0 1 2 3 4 5 6 7
0           . X X X X X X X
1           . . X X X X X X
2           . . . X X X X X
3           . . . . X X X X
4           . . . . . X X X
5           . . . . . . X X
6           . . . . . . . X
7           . . . . . . . .
```

The mask is `[8,8]`, shared by both examples and all eight heads because its rule depends on position, not sentence length. In head layout, decoder Q/K/V are `[2,8,8,32]`, scores are `[2,8,8,8]`, and masking is equivalent to:

```python
scores = scores.masked_fill(causal[None, None, :, :], float("-inf"))
weights = scores.softmax(dim=-1)
output = weights @ V
```

Consider example 0, query position 1. Its input is `V`, and it must predict `enez`. It may read BOS and `V` only. The training input already contains `enez` at position 2, but the mask hides that position. This prevents **answer leakage**. The diagonal remains allowed because the current input predicts the next token.

After merged heads and the output projection, the decoder does `h = ln_1(h + attention_output)`, shape `[2,8,256]`.

**There is no French key-padding mask in this implementation.** This is safe for this loss setup because padding is on the right: the graded query positions are 0-3 in example 0 and 0-6 in example 1. None can see a PAD input in its causal prefix. Queries with later PAD inputs can read earlier PADs, but all of those query outputs have ignored losses. That argument would not hold unchanged for left padding or sequence packing.

### 11. Mask application 3: cross-attention blocks English PAD again

The decoder now asks English memory for information. Cross-attention projects queries from the updated French vectors; keys and values come from English memory:

```text
French queries before head split       [2,8,256]
English keys and values                 [2,6,256]
Q in head layout                        [2,8,8,32]
K and V in head layout                  [2,8,6,32]
scores = Q @ K.transpose(-2,-1)/sqrt(32) [2,8,8,6]
```

**French length 8 and English length 6 do not need to match.** Each French position scores six English positions. The resulting value combination has one vector per French query, so its length is 8, not 6.

The actual call is:

```python
attn_out, _ = self.cross_attn(
    h, memory, memory,
    key_padding_mask=src_pad_mask,
    need_weights=False,
)
```

The `[2,6]` English padding mask expands to `[2,1,1,6]` again. There is no cross-attention causal mask: English is already fully known. The measured first decoder block, example 0, head 0, query position 0 (BOS after self-attention) gives:

| English key position | `Come` | `␠on` | `.` | EOS | PAD | PAD |
|---|---:|---:|---:|---:|---:|---:|
| Raw score | -0.829240 | -0.719274 | -0.306931 | -0.422521 | -0.506033 | -0.663632 |
| Masked score | -0.829240 | -0.719274 | -0.306931 | -0.422521 | negative infinity | negative infinity |
| Weight | 0.188536 | 0.210451 | 0.317855 | 0.283158 | **0** | **0** |

Thus this query's head output is:

```text
0.188536*V_Come + 0.210451*V_on + 0.317855*V_period + 0.283158*V_EOS
+ 0*V_PAD_4 + 0*V_PAD_5
```

Padded English memory vectors are generally nonzero. Encoder masking prevents valid English vectors from reading PAD, but it does not delete or zero the PAD query outputs. The decoder must mask those memory columns too. Otherwise it could read information through the padding slots.

The cross-attention head result is `[2,8,8,32]`. Merging heads and applying the output projection gives `attn_out` `[2,8,256]`, compatible with the French residual: `h = ln_2(h + attn_out)`.

### 12. Finish all decoder blocks

The feed-forward sublayer performs `256 -> 1024 -> GELU -> 256`, independently for all 16 batch/query positions, followed by residual addition and `ln_3`. The executed order in each block is:

```python
h = ln_1(h + french_causal_self_attention(h))
h = ln_2(h + english_cross_attention(h, memory, src_pad_mask))
h = ln_3(h + ff_out(gelu(ff(h))))
```

The same English memory is supplied to every decoder block; it is not re-encoded or modified by the decoder. Each block has its own projections and therefore can use that memory differently.

| Stage | Output shape |
|---|---|
| Decoder block 1 | `[2,8,256]` |
| Decoder block 2 | `[2,8,256]` |
| Decoder block 3 | `[2,8,256]` |
| Decoder block 4 | `[2,8,256]` |

The four blocks execute the same mask rules with independent learned weights. The model has no dropout, so switching it to evaluation mode does not disable any dropout in this trace.

### 13. Output projection: sixteen distributions over 16,000 tokens

The output projection is a linear layer without bias. Its weight is tied to the decoder token embedding: **it is the same parameter object**, shape `[16000,256]`.

```text
logits = decoder_hidden @ decoder_embedding_table.T
[2,8,256] @ [256,16000] -> [2,8,16000]
```

A **logit** is a raw category score. `logits[0,0,57]` is the score for `V` at example 0's first prediction. `logits[0,0,3926]` is a competing score for `enez` at that same position. IDs are category labels; their numerical distance is irrelevant. Predicting ID 56 is not inherently closer to target 57 than predicting ID 9000.

Sixteen score rows are produced, including those that will be ignored. This forward pass does not choose words with argmax, does not sample tokens, and does not print a translation. It supplies scores to the training loss. Although generation forbids PAD and BOS as outputs, the training cross-entropy still includes all 16,000 output categories in its probability denominator.

### 14. Mask application 4: make a loss-only expected-answer copy

Only after computing logits, the training loop runs:

```python
target = ids_to_matchable(tgt)
```

The helper is:

```python
rows = rows.clone()
rows[rows == 0] = -100
return rows
```

The original French data and the decoder input are not modified. Using descriptive aliases:

```text
expected_tokens = target                         shape [2,8]
row 0: [57, 3926, 779,   1,    -100, -100, -100, -100]
row 1: [38,   10, 344, 927,    4580,   17,    1, -100]
```

`-100` is an explicit ignore sentinel understood by cross-entropy. It is not a token, not a very distant category, and not a way of forcing a high loss. It never enters an embedding lookup, never gets its own output score, and is never used as an index into the vocabulary. **No loss calculation is performed for that position.**

The complete grading alignment is:

| Position | Example 0 decoder input | Example 0 expected output | Example 1 decoder input | Example 1 expected output |
|---|---|---|---|---|
| 0 | BOS | `V` | BOS | `C` |
| 1 | `V` | `enez` | `C` | `'` |
| 2 | `enez` | `␠!` | `'` | `est` |
| 3 | `␠!` | EOS | `est` | `␠mon` |
| 4 | EOS | Ignore | `␠mon` | `␠bureau` |
| 5 | PAD | Ignore | `␠bureau` | `.` |
| 6 | PAD | Ignore | `.` | EOS |
| 7 | PAD | Ignore | EOS | Ignore |

There are **4 + 7 = 11 valid targets** and **5 ignored targets**. EOS contributes a valid loss in each row. The prediction *after* EOS is not graded.

### 15. Flatten scores and calculate the scalar loss

The current call is:

```python
loss = F.cross_entropy(
    logits.float().reshape(-1, VOCAB_SIZE),
    target.reshape(-1),
)
```

PyTorch defaults are `ignore_index=-100`, `reduction="mean"`, and no label smoothing or class weights. Written explicitly, it is equivalent to:

```python
loss = F.cross_entropy(
    logits.float().reshape(16, 16000),
    target.reshape(16),
    ignore_index=-100,
    reduction="mean",
)
```

Flattening preserves row order: all eight positions from example 0, then all eight from example 1. The 16,000-category axis remains intact. In GPU training, `.float()` converts scores to FP32 for this calculation; the CPU trace already has FP32 scores.

For flat position `i`, let `z[i,c]` be a logit and `y[i]` its expected ID. Define the valid set `valid = {i : y[i] != -100}`. The mathematical loss is:

```text
log_probability_of_correct_token(i)
    = z[i,y[i]] - log(sum(exp(z[i,c]) for c in 0..15999))

loss = -sum(log_probability_of_correct_token(i) for i in valid) / len(valid)
```

Implementations evaluate the logarithmic normalization using numerically stable log-sum-exp, not a naive unbounded exponentiation. Ignored positions are excluded before looking up `y[i]` as a category, and from the averaging denominator. Do not manually apply softmax before `F.cross_entropy`; it expects logits and handles the normalization itself.

The measured **unreduced losses** from the seed-7 trace are:

```text
example 0:
[10.309103, 9.655269, 9.966477, 9.621701, 0, 0, 0, 0]

example 1:
[10.156885, 9.626239, 9.677705, 9.796798, 9.280848, 10.011780, 9.616413, 0]

sum of valid losses = 107.719223
valid target count  = 11
mean loss           = 107.719223 / 11 = 9.792656 nats
```

The zeros shown by `reduction="none"` are explicit placeholders for ignored positions, not evidence that PAD was predicted perfectly. Dividing the sum by **16** would be wrong: it would dilute the loss with padding. Taking an unweighted average of the two sentence means would also differ from the actual code, which gives equal weight to each valid token.

A uniform distribution over 16,000 categories would give `ln(16000) = 9.680344` per target. This random model is slightly worse, which is possible because its predictions are not uniformly distributed. Token accuracy is not the same as this loss: the loss measures probability assigned to the correct category, not just whether it ranks first.

### 16. End of the pass: what receives gradients?

The scalar loss completes the forward computation. The next training operations are:

```python
optimizer.zero_grad(set_to_none=True)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

The trace performs `backward` solely to verify that every ignored logit row has gradient zero; the measured maximum absolute gradient there is **0.0**. It does not execute `optimizer.step()`.

In real training, valid French losses propagate through the output projection, decoder, cross-attention, encoder, and their embeddings. The English encoder is trained indirectly through those French prediction losses; there is no separate English reconstruction loss. The two examples contribute to one combined scalar and one optimizer update.

One nuance: zero gradient for ignored output *positions* does not mean the entire PAD *embedding row* is frozen. Because the decoder embedding table is tied to the output projection, its PAD row can receive gradients as a competing output category at valid positions. `padding_idx` suppresses its embedding-lookup gradient, not every other use of that shared parameter.

## Masking reference: three attention applications plus one loss rule

This table summarizes what happened above. Reusing a source padding pattern is not creating a third kind of attention mask.

| Application | Mask shape for this batch | Applied to | Effect |
|---|---|---|---|
| 1. Encoder source-padding mask | `[2,6]`, broadcast as `[2,1,1,6]` | Encoder self-attention scores `[2,8,6,6]` | Block English PAD key columns before softmax |
| 2. Decoder causal mask | `[8,8]`, broadcast as `[1,1,8,8]` | French self-attention scores `[2,8,8,8]` | Block future French key columns before softmax |
| 3. Decoder source-padding mask | Same source pattern `[2,6]` | Cross-attention scores `[2,8,8,6]` | Block English PAD memory columns before softmax |
| 4. Loss ignore rule | Expected IDs `[2,8]`; ignore when `-100` | Cross-entropy over logits `[2,8,16000]` | Exclude five predictions and divide by eleven valid ones |

There are **two distinct attention-mask patterns applied in three places**, plus a loss-only ignore rule. The source masks differ per batch example; the causal rule is shared. The attention applications repeat in every applicable block. They do not shorten tensors or automatically remove query outputs.

For attention, in conceptual head layout:

```python
scores = Q @ K.transpose(-2, -1) / math.sqrt(head_width)
scores = scores.masked_fill(blocked_positions, float("-inf"))
attention_output = scores.softmax(dim=-1) @ V
```

For loss:

```python
valid = expected_tokens != -100
# Conceptual grading: only valid score rows and target IDs are used.
loss = F.cross_entropy(logits[valid], expected_tokens[valid])
```

The boolean mask convention above belongs to `MultiheadAttention`. Other APIs can use different boolean conventions. Never assume True means blocked without checking the API. If every key for a query were blocked, softmax over all negative infinities would be mathematically undefined; this pipeline retains nonempty content/EOS rather than relying on all-padding attention.

## Architecture reference

| Quantity | Configuration | This worked batch |
|---|---|---|
| Batch size | Normally 192 | 2 |
| Source / French input lengths | Up to 60 each; need not match | 6 / 8 |
| Model width | 256 | All hidden token vectors have 256 components |
| Encoder / decoder blocks | 4 / 4 | Four successive outputs of each stack |
| Heads per attention module | 8 | Head width 32 |
| Feed-forward width | 1,024 | Expand 256, bend with GELU, shrink to 256 |
| Vocabulary / output categories | 16,000 | Scores `[2,8,16000]` |
| Normalization / dropout | Post-LayerNorm / none | Same during this trace |
| Output tying | Decoder input/output weights shared | Encoder embeddings remain separate |
| Total parameters | 16,650,240 | Includes unused encoder cross-attention/norm parameters |

A Transformer block is larger than an attention module: it includes feed-forward and normalization/residual operations. Each attention module contains eight heads. Four blocks are four sets of learned transformations applied sequentially to the entire batch, not four portions of the sentence or four applications of identical weights.

## Data and tokenizer

[prepare_data.py](prepare_data.py) streams the `en-fr` training split of [OPUS-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100). It collects up to one million nonempty pairs, reserves the first 4,000 for validation, and uses the rest as training candidates. These counts are before length filtering.

A shared ByteLevel BPE tokenizer is trained on training text only. Byte-level pretokenization, its matching decoder, and a complete byte alphabet preserve spaces, punctuation, accents, and characters absent from the tokenizer-training examples.

| Token | ID | Role |
|---|---:|---|
| PAD | 0 | Fill unused row positions |
| EOS | 1 | Mark the end of English or French |
| BOS | 2 | Start French decoding |
| UNK | 3 | Reserved unknown symbol; byte coverage handles ordinary unseen characters |

Both sides must fit within **59 content tokens plus EOS**. If either side is too long, the whole pair is filtered out. The remaining sequences are right-padded to 60 and stored as `uint16`. Training loads them as integer tensors and trims unnecessary batch padding before the forward pass.

At inference, [pipeline.py](pipeline.py) rejects blank or overlength English input and appends EOS just as preparation does. Tokenizer hashes and format metadata reject incompatible artifacts. They do not fully fingerprint every training example or model setting.

## Generation is a different execution path from teacher-forced training

[translate.py](translate.py) receives English only. It tokenizes English, appends EOS, encodes it once, and starts French with BOS. At each generation iteration it runs the decoder on the prefix produced so far and uses only the last logit row. PAD and BOS scores are set to negative infinity, then greedy argmax selects the next token. EOS stops generation; no loss is calculated.

For example, if the model selected the correct pieces for `Venez !`, successive decoder inputs would be `[BOS]`, `[BOS,V]`, `[BOS,V,enez]`, `[BOS,V,enez,␠!]`. Choosing EOS after the final prefix would stop. That is an illustration of alignment, not a claim about a particular checkpoint's output.

Training instead supplies all known French prefixes in one causally masked forward pass and grades every valid position. The model's own mistakes can accumulate during generation, which is why low teacher-forced validation loss is not a guarantee of good translations. Generation has a 60-update limit and no beam search or key/value cache; the decoder recomputes each full prefix.

## Common myths, mistakes, and misunderstandings

A **mask** marks positions that a calculation must ignore. An attention mask restricts where information can be read; a loss mask excludes predictions from grading. They do not replace each other.

| Mistake or myth | Why it is wrong | What this implementation does |
|---|---|---|
| PAD tokens disappear when masked | Masking blocks key/value contributions, not tensor positions or padded queries | Keeps rectangular tensors and masks the right score columns |
| English and French must have equal token counts | Cross-attention scores have separate query and key lengths | Scores are `[2,8,8,6]` in this batch; outputs follow eight French query positions |
| `-100` creates a very high loss | It is an explicit ignore sentinel, not a category | Skips those positions and removes them from the mean denominator |
| Both `tgt` and `tgt_in` enter the decoder | The model call only receives `src` and `tgt_in` | `tgt` is copied separately into loss-only expected IDs |
| Zero PAD vectors make masking unnecessary | Nonzero score probabilities can still change normalization, and PAD vectors need not be zero | Explicitly masks scores before softmax |
| Ignore one sentence and the other sentence changes its attention | Batch entries are independent in attention | Only valid-token averaging combines their loss contributions |
| Mask source padding only in decoder cross-attention | Real English states have already absorbed padding through encoder self-attention | Applies `src == PAD` in every encoder self-attention block and every decoder cross-attention block |
| Assume `padding_idx` makes attention ignore padding | Position embeddings and later layers create nonzero states at padded positions; attention still needs a mask | Uses an explicit attention key-padding mask |
| Omit the decoder causal mask | A target position can read future French input, producing deceptively low training loss | Hides keys strictly after the current position; the diagonal stays visible |
| Treat loss masking as attention masking | Ignoring PAD targets only changes which predictions are graded | Handles attention masks and `-100` loss targets separately |
| Train `tgt[:-1] -> tgt[1:]` without a BOS prefix | The first French token is never a prediction target from a start context | Trains `BOS + target[:-1] -> target` |
| Start inference with EOS when training never used it as a start token | Generation begins from an unfamiliar context | Uses BOS consistently during training and inference |
| Ignore EOS in the loss | The model is not explicitly trained to stop | Only PAD is ignored; EOS contributes to cross-entropy |
| Forget English EOS during inference | The encoder sees a different input format from preparation | Appends EOS in the shared source encoder |
| Truncate both languages independently | The two fragments may stop at different meanings | Filters the whole pair if either side exceeds the limit |
| Use BPE without a matching text decoder | Subword pieces and spaces can be reconstructed incorrectly | Uses ByteLevel pretokenization and decoding together |
| Replace the tokenizer while retaining old weights | The same token IDs acquire different meanings | Checks tokenizer identity in data and checkpoints |

Decoder self-attention does not use a target key-padding mask here. With right padding and teacher forcing, every graded French position precedes the padding region, and the causal mask prevents it from reading later padding. Padded target positions have ignored losses. This reasoning would need revisiting for left padding or a different packing scheme.

With PyTorch `MultiheadAttention` boolean masks, **True means blocked**. Do not assume another attention API uses the same convention.

## Training, learning rate, and resume

A **loss** is a number measuring prediction error. **Cross-entropy** penalizes assigning low probability to the correct next token. **Backpropagation** computes the gradient: how the loss would change with each parameter. An **optimizer** uses those gradients to update the parameters. **AdamW** is the optimizer used here; it maintains moving estimates of gradients and their squared magnitudes and applies weight decay. The **learning rate**, abbreviated **LR**, scales its parameter updates. One **step** here means one optimizer update, not one entire pass through the dataset.

**Warmup** gradually raises LR at the start of training. **Gradient clipping** limits the combined gradient magnitude. **FP32**, or 32-bit floating point, and **BF16**, or brain floating point with 16 bits, are numerical formats; BF16 saves memory in many GPU operations while selected calculations remain in FP32. **CUDA** is the GPU execution backend used on supported NVIDIA hardware.


| Setting | Current value |
|---|---|
| Optimizer | AdamW; betas `(0.9, 0.95)`, epsilon `1e-8`; default weight decay |
| Batch size | 192, sampled with replacement |
| Gradient norm clipping | 1.0 |
| Peak LR | `5e-4` |
| Warmup | **4,000 absolute updates** |
| Evaluation / save interval | 10 updates, and the final update |
| Validation batch size | 128 |
| Precision | BF16 autocast on CUDA; otherwise FP32 |

For one-based absolute update `s`, the schedule is:

```python
lr = 5e-4 * min(s / 4000, (4000 / s) ** 0.5)
```

Warmup occurs once, then inverse-square-root decay continues without a fixed endpoint. `--steps` controls the number of additional updates, not the end of the LR schedule.

```powershell
python -m en2fr.train --resume --steps 5000
```

New checkpoints save model weights, AdamW state, completed step, schedule configuration, elapsed time, and NumPy/PyTorch random states. Resume restores these and appends logs. A saved schedule mismatch is rejected. Resuming an older format-2 checkpoint that contains weights but no optimizer/step requires the known completed update count:

```powershell
python -m en2fr.train --resume --start-step 5000 --steps 5000
```

That restores weights with a fresh optimizer and uses absolute step 5,001's LR. It cannot reconstruct the discarded optimizer history. Older incompatible data/tokenizer formats require rebuilding and retraining instead.

## Reading results and remaining limitations

Training loss is the latest random batch; validation loss is teacher-forced prediction loss on held-out text. Neither alone establishes free-running translation quality. Try unseen statements, questions, negation, accents, and unfamiliar names, checking both meaning and grammatical French.

Current limitations are deliberate or still unresolved:

- Validation uses the first examples from the source stream, without shuffling or deduplication. It is not a robust final benchmark.
- The validation loop drops an incomplete final batch and averages batch losses rather than weighting by valid-token count.
- Only the latest checkpoint is saved; there is no automatic best-validation checkpoint or early stopping.
- There is no automatic BLEU evaluation, beam search, dropout, or label smoothing.
- Sampling with replacement means a nominal dataset's worth of updates does not visit every pair once.
- BF16 capability is not checked separately from CUDA availability. Exact reproducibility across devices is not guaranteed.

## Files and tests

| File | Responsibility |
|---|---|
| [model.py](model.py) | Embeddings, encoder/decoder blocks, tied output projection |
| [pipeline.py](pipeline.py) | IDs, lengths, source encoding, artifact compatibility |
| [prepare_data.py](prepare_data.py) | Dataset, tokenizer, filtering, binary files |
| [train.py](train.py) | Optimization, validation, logging, checkpoint/resume |
| [translate.py](translate.py) | Greedy generation |
| [test_pipeline.py](test_pipeline.py) | Correctness and resume regression tests |
| [trace_batch.py](trace_batch.py) | Reproduce this two-example attention/loss walkthrough without changing checkpoints |

```powershell
python -m unittest -v en2fr.test_pipeline
```

Tests check padding invariance, tokenizer round-trip including Unicode, source EOS, empty/long input handling, whole-pair filtering, first-token gradients, absolute-step LR, checkpoint compatibility, resumed-update equivalence, and log/step continuity using temporary artifacts.
