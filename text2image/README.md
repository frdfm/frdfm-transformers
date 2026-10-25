# Caption -> grayscale clothing: a tiny pixel Transformer

A small image generator trained entirely from scratch on [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist). It reads a short caption and samples grayscale pixels, one at a time. The network learns directly from caption and pixel sequences. It does not use a pretrained image generator, compress images through another learned model, or retrieve existing pictures.

The scope is ten clothing categories at **28 x 28**, with a **14 x 14** option. These are simple product silhouettes, not a general text-to-image system. More resolution preserves details; it does not automatically improve prompt adherence.

## Start with a concrete picture

A **pixel** is one cell in an image. In a grayscale image, its value specifies brightness rather than color. Here each pixel can take one of **16 intensity levels**, from black to white. A 28 x 28 image contains 784 pixels.

A **token** is an item the Transformer reads as an integer **identifier**, or **ID**. This model has two kinds: caption words and pixel intensity values. A **vocabulary** assigns a distinct ID to each possible item. **PAD**, short for **padding**, is a special caption token meaning "unused slot"; it is not an image intensity.

For `draw a bag`, the input begins with IDs for `draw`, `a`, and `bag`, followed by enough padding slots to reach ten caption positions. Generation starts with only that caption. The model predicts the first pixel, appends it, predicts the second, and continues until it has all 784 pixels. This is **autoregressive generation**: each new output depends on earlier outputs.

Each ID is looked up in a learned table to produce an **embedding**, a vector of numbers. A **position embedding** adds information about the item's place in the sequence. The **Transformer** repeatedly combines these vectors through **attention**, which learns how strongly each position should use information from other permitted positions. **Causal** means future positions are hidden, so the model cannot read pixels it is supposed to predict.

The values learned during training are **parameters**. A **checkpoint** is a saved snapshot of these values and, for resuming training, the optimizer's state. A **batch** is a group of examples processed together. An **optimizer** is the algorithm that changes parameters to reduce prediction error.

## Run it

From the repository root with your venv active; dependencies are `torch`, `numpy`, and `Pillow`:

```powershell
python -m text2image.prepare_data --size 28
python -m text2image.train --size 28 --steps 5000
python -m text2image.generate --size 28 "draw a bag"
python -m text2image.generate --size 28 --all
```

Skip preparation if matching prepared data already exists. Continue training with:

```powershell
python -m text2image.train --size 28 --resume --steps 5000
```

`--steps` means additional updates. Without `--resume`, training replaces that resolution's log and checkpoint files. The model constructor defaults to 14 for old-checkpoint compatibility; the command-line tools default to **28**.

## Worked forward pass: a batch of two real images, through to loss

This section follows two real images through the **28 x 28 model**, using their actual prepared pixels and exact caption IDs. The training default is batch size 64; this walkthrough uses two so that the indexing and masks can be inspected. A batch is just a group of examples processed together; the attention calculation does not mix examples.

### Reproduce the trace

```powershell
python -m text2image.trace_batch
```

[trace_batch.py](trace_batch.py) selects the first bag and first coat in the prepared `data/28/train.npz`. In the current seeded split, these are rows **20 and 0, zero-based**. It constructs the captions below, initializes a new model with PyTorch seed 7, and runs on CPU in 32-bit floating point.

**It never loads or changes your trained checkpoint and performs no optimizer update.** It runs in evaluation mode to disable dropout, checks an explicit attention calculation against the PyTorch module, computes the actual loss, and performs a backward check of which output positions receive gradients. Model parameters still require gradients in evaluation mode: `eval()` disables training-specific behavior, not differentiation.

Numerical values below are measured from that trace, rounded for readability. They illustrate the implemented calculation, not the behavior or quality of a trained checkpoint. Small numerical differences across hardware or software are possible.

### 1. Start with images and their captions

| Batch example | Prepared training row | Fashion-MNIST class label | Caption chosen for this trace |
|---|---:|---:|---|
| 0 | 20 | 8, bag | `draw a bag` |
| 1 | 0 | 4, coat | `a grayscale image of a coat` |

Both captions use templates that actual training chooses randomly. The dataset supplies category labels, not human-written descriptions. A label selects its category name to build the caption; the class label itself is **not** passed into the Transformer as an extra input.

An image is an array of brightness values. Preparation preserves native 28 x 28 resolution and quantizes brightness into 16 levels:

```text
pixel_ID = round(original_brightness / 255 * 15)
possible IDs = 0,1,...,15
```

Zero means black in the stored image; 15 means white. Generation display later reverses this direction to put dark clothing on a white background. Display inversion is not part of the training tensors.

### 2. Flatten real image pixels, without padding them

Each image becomes a length-784 vector in row-major order: left to right across row 0, then row 1, and so on. Using zero-based image coordinates:

```text
pixel index 0   -> image row 0, column 0
pixel index 27  -> image row 0, column 27
pixel index 28  -> image row 1, column 0
pixel index 783 -> image row 27, column 27
```

Let `p[b,j]` denote pixel `j` in batch example `b`. The pixel tensor is `[2,784]`, a 64-bit integer tensor after loading. A shape such as `[2,784]` means two examples, each with 784 entries.

Actual data excerpts:

```text
Bag, p[0,0:28]:
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

Bag, p[0,38:50]:
[0,0,0,4,9,8,9,6,0,0,0,0]
         ^ first nonzero pixel: index 41, row 1, column 13

Coat, p[1,0:28]:
[0,0,0,0,0,0,0,0,0,0,6,9,11,11,10,11,11,10,3,0,0,0,0,0,0,0,0,0]
                     ^ first nonzero pixel: index 10, row 0, column 10
```

Both examples have last pixel `p[b,783] = 0`. The bag contains 312 zero pixels and the coat 305. Those are **real targets**, not unused slots. Since both images have the same dimensions, no image padding or pixel-length mask is needed.

### 3. Tokenize the two captions and right-pad to ten positions

Here a **token** is either a caption word or a pixel-intensity ID. A vocabulary is the fixed mapping from these items to integers. The input vocabulary contains 34 items: 16 intensities, a separate padding marker, and 17 words.

Relevant word IDs are:

```text
a=17, bag=19, coat=21, draw=22,
grayscale=25, image=26, of=27
PAD=16
```

The tokenizer lowercases captions, extracts alphabetic words, looks them up, and appends PAD IDs to reach `PREFIX = 10`:

```python
captions = torch.tensor([
    [22, 17, 19, 16, 16, 16, 16, 16, 16, 16],
    [17, 25, 26, 27, 17, 21, 16, 16, 16, 16],
])  # shape [2,10]
```

Example 0 has three real words and seven padding slots. Example 1 has six real words and four padding slots. We do not trim those captions to their real lengths: their fixed ten-position width establishes the same image-start boundary for every example.

**PAD is 16, not 0.** Treating zero as padding here would incorrectly mask genuine black image pixels. Token IDs for words are distinct from pixel IDs even when the same shared input embedding table stores their vectors.

### 4. Build the teacher-forced input sequence

Inside [train.py](train.py), `pixel_loss` creates:

```python
inputs = torch.cat([captions, pixels[:, :-1]], dim=1)
```

The input contains the entire caption plus the first 783 pixels, not all 784:

```text
inputs shape: [2,793] = [2,10 + 783]

positions 0..9: caption IDs
position 10:   p[b,0]
position 11:   p[b,1]
...
position 792:  p[b,782]
```

The final image pixel, `p[b,783]`, is a target but never an input in this pass. Earlier pixels are the known correct training-image values: this is **teacher forcing**. Supplying them allows all predictions to be calculated in one forward pass, provided future positions are masked.

For the actual nonzero pixels above:

| Example | Pixel target | Input position where that pixel appears | Logit position that must predict it |
|---|---|---:|---:|
| Bag | `p[0,41] = 4` | `10 + 41 = 51` | `9 + 41 = 50` |
| Coat | `p[1,10] = 6` | `10 + 10 = 20` | `9 + 10 = 19` |

The predicting position is one place before the input containing the answer. The causal mask must block that answer position. Pixel targets are the original `pixels` tensor; it is not a second argument to the Transformer:

```python
all_logits = model(inputs)  # Only inputs enters ImageTransformer.forward.
```

### 5. Input embeddings and positional embeddings

An **embedding** is a learned vector looked up using an ID. A **position embedding** adds a vector identifying the item's sequence position. Inside [model.py](model.py):

```python
length = tokens.shape[1]  # 793
x = self.embed(tokens) + self.position.weight[:length]
```

| Quantity | Shape |
|---|---|
| Input IDs | `[2,793]` |
| Token embedding parameters | `[34,128]` |
| Looked-up token vectors | `[2,793,128]` |
| Position embedding parameters | `[794,128]` |
| Position slice used in training | `[793,128]` |
| Sum `x` | `[2,793,128]` |

The position slice broadcasts across both batch examples. For example:

```text
x[0,2,:]  = embedding[19,:] + position[2,:]    # bag word
x[1,2,:]  = embedding[26,:] + position[2,:]    # image word
x[0,9,:]  = embedding[16,:] + position[9,:]    # PAD query slot
x[0,10,:] = embedding[0,:]  + position[10,:]   # first black pixel
```

Each resulting vector has 128 learned numbers. Equal IDs share their token embedding but not their position embedding. The two examples remain separate; sharing parameters does not mix their contents.

There is no `padding_idx` on this embedding table. Padding vectors exist and are trainable. As the next steps show, the network blocks them as attention keys/values while retaining their query states.

### 6. Construct two attention masks for the full sequence

`ImageTransformer.forward` creates both masks before entering the blocks:

```python
causal = torch.ones(length, length, dtype=torch.bool).triu(1)
padding = tokens == PAD  # Passed as this expression inside the loop.
```

**Mask 1: causal mask**, shape `[793,793]`. For query position `i`, key position `j` is blocked whenever `j > i`. The same pattern applies to both examples and every head.

**Mask 2: caption-padding mask**, shape `[2,793]`. It blocks key positions whose input ID is 16. For this batch:

```text
example 0, first ten slots: [F,F,F,T,T,T,T,T,T,T]
example 1, first ten slots: [F,F,F,F,F,F,T,T,T,T]
all 783 pixel-input slots: False in both examples
```

Here `T` means boolean True, blocked. Unlike some other APIs, boolean True means blocked in PyTorch `MultiheadAttention`. The padding mask does not shorten the sequence or delete query positions.

Selected attention rows show how the masks combine:

| Example and query | Allowed keys after applying both masks | Why other positions are blocked |
|---|---|---|
| Bag, position 9 | Caption positions 0,1,2 only | 3-9 are PAD keys; 10-792 are future pixels |
| Coat, position 9 | Caption positions 0-5 only | 6-9 are PAD keys; 10-792 are future pixels |
| Bag, position 10 | Caption 0,1,2 plus pixel input position 10 | PAD keys blocked; later pixels hidden |
| Coat, position 10 | Caption 0-5 plus pixel input position 10 | PAD keys blocked; later pixels hidden |
| Bag, position 50, predicting pixel 41 | Real caption words and pixel input positions 10-50 | Position 51, containing the target value 4, is future |

**The diagonal is allowed by the causal rule**, but a diagonal PAD key is still blocked by the padding rule. A padded query can nevertheless attend to earlier real keys. No query in this batch has all keys blocked, since every caption starts with real words.

### 7. First processing block: normalize, then project queries/keys/values

There are four independent blocks. Each keeps the complete `[2,793,128]` hidden state. The first operation of a block is:

```python
h = self.norm1(x)
```

**LayerNorm**, layer normalization, computes mean and variance across the 128 components of each individual vector. It does not normalize across examples or positions:

```text
normalized_j = gamma_j * (x_j - mean(x)) / sqrt(variance(x) + 1e-5) + beta_j
```

Variance here is the mean squared deviation over those 128 components. Scale `gamma` and offset `beta` are learned. This is **pre-LayerNorm**: normalization precedes each attention/feed-forward sublayer, unlike the translator's post-LayerNorm ordering.

Attention projects `h` into queries, keys, and values with three learned linear maps. A linear map computes `h @ weight.T + bias`. PyTorch packs these maps into `in_proj_weight` `[384,128]` and `in_proj_bias` `[384]`; each slice is `[128,128]` with a 128-element bias.

```text
Q = linear_query(h)   [2,793,128]
K = linear_key(h)     [2,793,128]
V = linear_value(h)   [2,793,128]
```

A **query** is used to look for matching keys; **values** carry the information to combine after matching. Since all three inputs come from this same sequence, it is **self-attention**. There is no separate encoder or cross-attention.

Four heads rearrange those vectors into a conceptual layout `[2,4,793,32]`: two examples, four heads, 793 positions, 32 values per head. Raw scores are:

```python
scores = Q @ K.transpose(-2, -1) / math.sqrt(32)
# shape [2,4,793,793]
```

The last two dimensions are query position and key position. The batch axis is never summed across: a bag query cannot read the coat image or caption.

### 8. Apply both masks before softmax and value aggregation

In the conceptual head layout, equivalent expanded code is:

```python
scores = scores.masked_fill(padding[:, None, None, :], float("-inf"))
scores = scores.masked_fill(causal[None, None, :, :], float("-inf"))
weights = scores.softmax(dim=-1)
head_output = weights @ V
```

Padding broadcasts as `[2,1,1,793]`; causality broadcasts as `[1,1,793,793]`. A position blocked by either rule receives negative infinity before softmax. Because `exp(-infinity)=0`, its weight is zero, its value contributes nothing, and its raw score does not contribute to the normalization denominator.

A measured first-block, head-0 row for **bag query position 9**, which predicts pixel 0:

| Key position | 0: draw | 1: a | 2: bag | 3-9: caption PAD | 10: pixel 0 | 11: pixel 1 | Later keys |
|---|---:|---:|---:|---|---:|---:|---|
| Raw score | -0.101361 | -0.653641 | -0.962744 | Finite values | 0.991292 | 0.360889 | Finite values |
| After masking | -0.101361 | -0.653641 | -0.962744 | negative infinity | negative infinity | negative infinity | negative infinity |
| Weight | **0.500447** | **0.288075** | **0.211478** | **0** | **0** | **0** | **0** |

Even though future pixel 0 has a larger raw score than every caption word here, the causal mask removes it before softmax. Its value cannot leak into its own prediction.

The resulting 32-number head vector is:

```text
0.500447 * V_draw + 0.288075 * V_a + 0.211478 * V_bag
+ zero times every blocked value vector
```

For the coat at query position 9, measured weights over its six real caption keys are:

```text
[a, grayscale, image, of, a, coat]
[0.196480, 0.095437, 0.085071, 0.150865, 0.244358, 0.227789]
```

For bag query position 10, the allowed keys now include pixel 0, because it is predicting pixel 1:

```text
key position:       0         1         2         10
meaning:            draw      a         bag       pixel 0
attention weight:   0.245655  0.262908  0.213901  0.277535
```

All other weights are zero. Pixel 0's intensity ID is 0, but it is not padding and receives nonzero attention. The trace asserts that all blocked weights are exactly zero and verifies the expanded result against the actual module call:

```python
self.attention(h, h, h,
               attn_mask=causal,
               key_padding_mask=padding,
               need_weights=False)
```

`need_weights=False` means the module does not return its attention weights; it does not disable masking or attention. Optimized attention may fuse calculations rather than materialize the full conceptual score array.

### 9. Finish block 1 and process blocks 2-4

Per-head outputs have shape `[2,4,793,32]`. Merge the heads to `[2,793,128]`, then apply the learned output projection `[128,128]` plus bias `[128]`.

The complete block order is:

```text
incoming x                                            [2,793,128]
h = norm1(x)                                          [2,793,128]
a = masked_attention(h,h,h)                           [2,793,128]
x = x + a                                             [2,793,128]
h = norm2(x)                                          [2,793,128]
f = Linear(128 -> 512)(h)                              [2,793,512]
f = GELU(f)                                           [2,793,512]
f = Linear(512 -> 128)(f)                              [2,793,128]
f = Dropout(f)                                        [2,793,128]
block_output = x + f                                  [2,793,128]
```

A **residual addition** adds a learned correction to its incoming vector. **GELU**, Gaussian error linear unit, supplies nonlinearity; its default exact formula is `0.5*x*(1 + erf(x/sqrt(2)))`. The feed-forward operations act independently on each position; attention is the operation that mixes information across positions.

During normal training, attention weights and feed-forward output use dropout probability 0.1. That randomly zeroes selected activations and scales retained ones by `1/(1-0.1)`. Attention masking still blocks keys first; dropout does not reintroduce them, though post-dropout attention weights need not sum to one. The numerical trace disables dropout using `eval()` to make comparisons reproducible.

All four blocks receive the same mask patterns but have **separate learned parameters**:

| Stage | Output shape |
|---|---|
| Block 1 | `[2,793,128]` |
| Block 2 | `[2,793,128]` |
| Block 3 | `[2,793,128]` |
| Block 4 | `[2,793,128]` |

Block 2 works on block 1's vectors, not the original token IDs. Neither caption nor pixel arrays are split between blocks. Each attention module has four heads; a head and a block are not the same unit.

### 10. Final normalization and sixteen output categories

A final LayerNorm is applied after block 4, producing `[2,793,128]`. The output linear layer has weights `[16,128]` and bias `[16]`:

```text
all_logits = final_hidden @ output_weight.T + output_bias
[2,793,128] @ [128,16] -> [2,793,16]
```

A **logit** is a raw score before conversion into probability. At any position there are 16 scores, one per candidate next-pixel intensity. There are **not** 34 output categories: the model reads 34 types of input tokens but only predicts pixel values 0-15. It cannot emit caption words or PAD=16 through this output head. Unlike the translator, its input embedding and output projection weights are not tied.

The forward pass returns all 793 score rows per example. The loss function, outside `ImageTransformer.forward`, decides which rows to grade.

### 11. Output selection: why the loss slice starts at 9

The training code selects:

```python
logits = all_logits[:, PREFIX - 1:]  # PREFIX=10, so start at position 9
```

This keeps 784 positions: `793 - 9 = 784`. Each output is aligned with one original image pixel:

| Input sequence position | Input item at that position | Prediction target | Included in loss? |
|---|---|---|---|
| 0-8 | Real caption words or caption PAD | No pixel prediction is graded here | No: slice excludes these rows |
| 9 | Final caption PAD, in both examples | Pixel 0 | **Yes** |
| 10 | Pixel 0 | Pixel 1 | Yes |
| 11 | Pixel 1 | Pixel 2 | Yes |
| 19 | Pixel 9 | Pixel 10; coat target is **6** | Yes |
| 50 | Pixel 40 | Pixel 41; bag target is **4** | Yes |
| 792 | Pixel 782 | Pixel 783; target is **0** in both examples | Yes |

The selected logits have shape `[2,784,16]`, matching targets `[2,784]`. Starting at index 10 would discard pixel 0 and misalign all targets. Slicing off ten positions rather than nine is an off-by-one bug.

**Why is a PAD query graded?** The mask blocks padding *keys/values*. Position 9 still has a hidden state and a query. It reads real caption words and has its own learned token/position vectors; it acts as the fixed boundary from caption to image. Its output must predict the first pixel, so it participates in loss. The PAD token is not the desired output: the target at this position is the real pixel ID 0.

This is different from the translator, where predictions whose French targets are PAD are ignored. A token's presence as a padded **input** does not, by itself, determine whether its output is graded; the target alignment does.

### 12. Flatten and calculate cross-entropy over all 1,568 pixels

The exact training calculation is:

```python
loss = F.cross_entropy(
    logits.reshape(-1, logits.shape[-1]),
    pixels.reshape(-1),
)
```

For this batch it is equivalent to:

```python
loss = F.cross_entropy(
    logits.reshape(1568, 16),
    pixels.reshape(1568),
    reduction="mean",
)
```

Flattening retains example order: the bag's 784 output/target pairs, then the coat's 784. It does not merge the 16-category axis with position. `pixels` contains only IDs 0-15. Although PyTorch's default ignore sentinel is `-100`, **there are no `-100` targets here**, so none are ignored. There is no conversion of zero into `-100`.

For example `b`, target pixel `j`, and intensity category `c`, define `z[b,j,c] = all_logits[b,9+j,c]`. The per-pixel loss and batch mean are:

```text
pixel_loss[b,j] = -z[b,j,pixels[b,j]]
                 + log(sum(exp(z[b,j,c]) for c in 0..15))

batch_loss = sum(pixel_loss[b,j] for both b and all 784 j) / 1568
```

PyTorch uses a numerically stable log-sum-exp calculation. Supply logits directly; do not manually softmax them before calling cross-entropy. IDs identify categories: an intensity 3 prediction is not penalized by numerical distance from target 4. Cross-entropy grades the probability assigned to category 4.

Measured values from the seed-7 random-model trace:

| Quantity | Bag | Coat |
|---|---:|---:|
| First pixel's target | 0 | 0 |
| First pixel's loss | 2.961888 | 2.732293 |
| Second pixel's loss | 1.919113 | 1.813122 |
| Loss at first nonzero pixel | 4.101198, target 4 at index 41 | 3.130738, target 6 at index 10 |
| Sum of all 784 pixel losses | 2083.194824 | 2093.285156 |
| Per-image mean | 2.657136 | 2.670007 |

```text
total loss sum = 4176.479980
number graded  = 2 * 784 = 1568
batch loss     = 4176.479980 / 1568 = 2.663571 nats per pixel
```

The two per-image sums may differ from the combined sum in the last digits due to floating-point summation order. Each image has the same number of valid pixels, so an average of the two image means matches the batch mean within rounding. This differs from variable-length French sentences, where equal weighting of sentence means would change the loss.

All **617 zero-valued pixels** in these two images contribute to that calculation. Ignoring zero would throw away real background/content information and create a different, incorrect training objective. Uniform probability over sixteen intensities gives `ln(16) = 2.772589`; this particular randomly initialized model does slightly better on this particular batch by chance. That does not establish learned image-generation ability.

### 13. End of the forward pass, and what backward confirms

The result is one scalar loss. In actual training, the next operations are zeroing old gradients, backpropagation, clipping the combined gradient norm to 1.0, and one AdamW optimizer update. No sampling, argmax, or PNG rendering occurs inside this forward/loss computation.

The read-only trace additionally checks gradients without updating parameters:

| Check | Measured result | Interpretation |
|---|---:|---|
| Maximum absolute gradient of output logit rows 0-8 | **0.0** | Those rows are not selected by the loss slice |
| Gradient norm of logit row 9 across the batch | 0.00088749 | The first-pixel prediction is graded |
| Gradient norm of embedding row PAD=16 | 0.00048497 | A masked key token can still receive gradient through its query/residual role |

Zero gradient on a discarded *logit row* does not mean every underlying caption hidden state has zero gradient. Real caption representations influence later image predictions as attention keys/values, so the loss trains their embeddings and earlier block states indirectly. Caption PAD positions are blocked as keys at every block, but position 9 remains a directly graded query. This is why "mask PAD" and "freeze every PAD-related parameter" are different statements.

## Masking and alignment reference

There are **two attention masks and one output-selection rule** here, not the translator's three attention applications plus target-ignore rule.

| Rule | Shape for this batch | When applied | What it excludes |
|---|---|---|---|
| Caption key-padding mask | `[2,793]`, broadcast `[2,1,1,793]` | Every block's attention scores, before softmax | PAD=16 keys/values; never black pixel ID 0 |
| Causal mask | `[793,793]`, broadcast `[1,1,793,793]` | Every block's attention scores, before softmax | Future sequence positions |
| Loss score slice `[:,9:]` | `[2,793,16]` becomes `[2,784,16]` | After the full model forward pass | Output score rows 0-8, not pixel targets |
| Target ignore sentinel | Not used in these image targets | No filtering occurs | Nothing; all 1,568 pixels are graded |

Masks do not delete tensor positions. The final caption position remains a query even though it is PAD. Caption lengths can differ, image sizes in a batch cannot, and attention never joins different batch examples. The same mask patterns are reused by all four independently parameterized blocks.

## Generation: same position alignment, different source of earlier pixels

At inference, only the caption is known. Both examples could be sampled together:

| Sampling iteration | Model input shape | Logits shape | Score row used | New result |
|---|---|---|---:|---|
| 1 | `[2,10]` | `[2,10,16]` | 9 | Sample pixel 0 for each example |
| 2 | `[2,11]` | `[2,11,16]` | 10 | Sample pixel 1 |
| 3 | `[2,12]` | `[2,12,16]` | 11 | Sample pixel 2 |
| 784 | `[2,793]` | `[2,793,16]` | 792 | Sample pixel 783, then stop |

The next pixel is sampled from `softmax(last_logits / temperature)`. No expected-answer tensor or loss is needed for generation. After 784 samples, `[2,784]` is reshaped into `[2,28,28]`. Image dimensions provide the stopping rule: there is no beginning/end-of-sequence token for images in this implementation.

Teacher-forced training uses correct earlier pixels; generation uses its own earlier samples. Errors can accumulate along the image even if validation loss falls. The sampler recomputes its full prefix each time; it has no key/value cache for reusing earlier attention projections. PNG saving reverses brightness for display and enlarges pixels without adding resolution.

## Architecture and size reference

| Setting | 14 x 14 | 28 x 28 |
|---|---:|---:|
| Pixel targets per image | 196 | 784 |
| Fixed caption slots | 10 | 10 |
| Training input positions | 205 | 793 |
| Input vocabulary / output categories | 34 / 16 | 34 / 16 |
| Blocks / width / attention heads | 4 / 128 / 4 | 4 / 128 / 4 |
| Width per head / feed-forward intermediate width | 32 / 512 | 32 / 512 |
| Parameters | 826,128 | 901,392 |
| Measured peak allocated GPU memory, batch 64 | 0.68 GiB | 4.53 GiB |

Both use pre-LayerNorm, a final LayerNorm, and untied input/output weights. Doubling image dimensions produces four times as many pixels and roughly sixteen times the attention-score work. Memory figures are short training measurements on an RTX 3060 12 GB, not universal guarantees; total device usage can be higher.

## Vocabulary and prompts

Pixel tokens are IDs **0-15**. Caption PAD is **16**. Word IDs start at **17**, in a fixed vocabulary built from the category names and templates. In particular, **pixel zero is real black image content, not padding**.

Captions are lowercased and split with an alphabetic regex; punctuation and hyphens are separators. Unknown words, empty prompts, and prompts over ten words are rejected. The parser does not validate whether a combination of known words is meaningful.

| Category names |
|---|
| `t-shirt`, `trousers`, `pullover`, `dress`, `coat` |
| `sandal`, `shirt`, `sneaker`, `bag`, `ankle boot` |

Each training example receives one randomly selected synthetic caption template:

```text
draw a {category}
a drawing of a {category}
a grayscale image of a {category}
{category}
```

Use those forms. The model has not learned arbitrary scene composition, color, multiple objects, or attributes such as striped. All text embeddings are learned locally, but ten labels and synthetic captions provide limited language supervision.

## Data preparation

[prepare_data.py](prepare_data.py) downloads about 30 MB of compressed IDX files from the Fashion-MNIST authors' repository, verifies the published MD5 checksums, and caches the archives under `data/`. The upstream dataset is MIT licensed.

The native images are 28 x 28. At size 28, their resolution is preserved; at size 14, adaptive average pooling averages 2 x 2 regions. Intensities are quantized into 16 levels. A seeded shuffle of the original 60,000 training examples produces 55,000 training and 5,000 validation examples. The official 10,000 test images are prepared separately and are not used by the training loop.

Pixels and category labels are stored in compressed NumPy files. Captions are generated during training rather than stored as natural human descriptions. Model weights start randomly initialized.

## Training and learning rate

**Loss** measures prediction error. Here **cross-entropy** penalizes low probability for the correct pixel intensity. **Gradients**, computed through backpropagation, describe how changing each parameter affects loss. **AdamW** is the optimizer that uses these gradients to update parameters while maintaining moving estimates of gradients and squared gradients. **Weight decay** additionally shrinks weights. **LR** means **learning rate**, the scale of each optimizer update. One training **step** is one such update.

**Warmup** raises LR gradually at the start. **Gradient clipping** limits the combined gradient magnitude. **FP32** means 32-bit floating-point arithmetic. **TF32**, or TensorFloat-32, is a lower-precision internal format allowed for some GPU matrix multiplications here; model parameters remain FP32. **CUDA** is the execution backend for supported NVIDIA GPUs, and **GiB** means gibibytes of memory, where one GiB is 1,073,741,824 bytes.


| Setting | Default |
|---|---|
| Optimizer | AdamW, LR peak `3e-4`, weight decay `0.01`, default betas |
| Batch size | 64, sampled with replacement |
| Gradient clipping | Norm 1.0 |
| Warmup | 500 absolute updates |
| Evaluation / preview / checkpoint | Every 250 updates and at the final update |
| Validation | All 5,000 images; fixed `draw a ...` captions |
| Precision | FP32; CUDA matmul TF32 allowed |

For absolute one-based update `s`:

```python
lr = 3e-4 * min(s / 500, (500 / s) ** 0.5)
```

Resume restores weights, optimizer, step, best validation loss, and PyTorch random states. Warmup and decay do not restart. Preview sampling uses isolated random states so creating previews does not change subsequent training randomness. The image checkpoint does not currently validate a saved LR-schedule configuration; keep training constants consistent when resuming.

Useful options:

```powershell
python -m text2image.train --size 28 --steps 5000 --batch-size 32
python -m text2image.train --size 28 --resume --steps 5000 --no-samples
```

`--val-batches N` limits evaluation to a fixed prefix of validation batches; `0` means all. Keep the evaluation setting consistent when comparing runs. A two-step plumbing test is possible with `--steps 2 --batch-size 2 --eval-every 1 --val-batches 1 --no-samples`, but it writes the normal checkpoint path, so do not use it over a valued trained run.

## Generation and artifacts

```powershell
python -m text2image.generate --size 28 "a drawing of a sneaker" --count 8 --temperature 0.8 --seed 12
python -m text2image.generate --size 28 --all --seed 123
```

**Softmax** converts scores into probabilities that sum to one. Sampling uses `softmax(logits / temperature)`, then draws the next intensity according to those probabilities. Temperature must be positive; lower values sharpen choices and often reduce diversity. A fixed seed makes checkpoint comparisons easier, but a single seed is not a quality benchmark. `--all` generates four examples per category. Use `--output path.png` to choose a destination.

The model trains on the original intensity direction. Display reverses intensities to put dark shapes on white and enlarges pixels using nearest-neighbor scaling. Enlargement adds no image detail.

| Artifact | 14 x 14 path | 28 x 28 path |
|---|---|---|
| Prepared splits | `data/{train,val,test}.npz` | `data/28/{train,val,test}.npz` |
| Latest checkpoint | `checkpoints/model.pt` | `checkpoints/28/model.pt` |
| Best validation checkpoint | `checkpoints/best.pt` | `checkpoints/28/best.pt` |
| CSV log | `logs/train.csv` | `logs/28/train.csv` |
| Training previews | `samples/step-NNNNNN.png` | `samples/28/step-NNNNNN.png` |
| Default generated grid | `samples/generated.png` | `samples/28/generated.png` |

Paths are relative to this folder. Generation loads the **latest** `model.pt`, not `best.pt`; there is no checkpoint-selector flag. A saved model config records resolution; older configs without it default to 14. Existing size-14 models still work with `--size 14`. Switching resolution requires preparing matching data and training a new model: position embeddings have different sizes.

## Common myths, mistakes, and how to read results

A **mask** marks positions excluded from a calculation. A **causal mask** hides future sequence positions; a **padding mask** hides unused caption slots as attention keys. Those keys are vectors used to decide which positions to read. Neither mask means "ignore black pixels."

| Mistake | Consequence / correct approach |
|---|---|
| Every padded input must have an ignored output loss | Position 9 is a PAD query that predicts a real pixel and is intentionally graded |
| Blocking PAD keys freezes the PAD embedding | Queries/residual states are still computed; this trace measures nonzero PAD embedding gradients |
| Captions must have the same real word count | Right padding gives a shared ten-slot prefix; the mask blocks different key slots per example |
| The target image is a second model input | Only caption plus shifted pixels enters the model; full pixels separately enter the loss |
| There must be 34 output categories because there are 34 input tokens | The independent output head predicts only 16 pixel intensities |
| Reuse the translator's `0 -> -100` target conversion | Image zero is valid content, not padding; all image targets are graded |
| Treat pixel ID 0 as padding, as in the translator | Removes real image content; only ID 16 is caption PAD |
| Omit causal masking | Future training pixels leak into predictions; generation cannot reproduce the apparent validation performance |
| Forget the caption padding mask | Padding positions become usable content keys; every block masks them explicitly |
| Slice logits starting at position 10 instead of 9 | Drops the first pixel prediction and shifts alignment |
| Include caption positions in pixel cross-entropy | Trains a different objective; only the P aligned image logits are graded |
| Add an EOS requirement without corresponding training targets | The model never learned that stopping rule; current generation stops by pixel count |
| Change vocabulary order or quantization and reuse old weights | Token meanings change; regenerate compatible data and retrain |
| Interpret fixed-seed repeated previews as proof of collapse | Compare multiple fresh samples per category to assess diversity |
| Assume a lower pixel loss guarantees recognizable objects | Easy background and neighboring-pixel predictions can dominate; inspect category adherence and whole-image structure |
| Assume more pixels solve malformed objects | Higher resolution increases the modeling burden and preserves detail, but does not enforce coherent shapes |
| Compare 14 x 14 and 28 x 28 losses as identical tasks | Pooling changes the pixel distribution and sequence length; evaluate each resolution on its own images |

Uniform prediction over 16 intensities gives cross-entropy `ln(16)`, about **2.77 nats per pixel**. A randomly initialized model can be worse than this because its probabilities need not be uniform. Zero would mean perfect, completely confident predictions; it is not a practical target for a stochastic image distribution.

Training loss is one sampled batch with dropout; validation averages examples in evaluation mode. Train loss can therefore exceed validation loss without indicating a bug. The observed 28 x 28 run reached validation loss **0.7493 at step 5,000**; that is one run's result, not a promised target.

The generated samples show that plausible local pixels can still form an incorrect object, disappear into background, or lose structure as autoregressive errors accumulate. Check recognizable shapes, correct category, diversity, and failure rate over multiple seeds. Neither a loss curve nor one attractive preview establishes all four. No automated perceptual, category-accuracy, or memorization evaluation is implemented.

## Files and tests

- [model.py](model.py): text vocabulary, pre-LN blocks, causal forward pass, pixel sampling.
- [prepare_data.py](prepare_data.py): verified downloads, split, pooling, quantization.
- [train.py](train.py): teacher forcing, validation, optimizer, resume, previews.
- [generate.py](generate.py): prompt CLI and PNG grids.
- [test_model.py](test_model.py): prompt handling, causality, alignment, gradients, resolutions, sampling, PNGs.

- [trace_batch.py](trace_batch.py): reproduce the two-real-image forward pass, explicit attention checks, loss, and gradient checks without changing checkpoints.

```powershell
python -m unittest -v text2image.test_model
```

Pixel-sequence modeling is inspired by [Image GPT](https://github.com/openai/image-gpt). This is an independent small educational implementation, not a reproduction of that model's architecture or results.
