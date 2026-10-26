# Stick figures generated as pixels

**This is the pixel-based experiment.** The Transformer receives text and earlier image pixels, and predicts the next pixel. It outputs no joint coordinates, body-part angles, or drawing commands. Inference performs no skeleton rendering and does not retrieve a reference drawing.

The earlier [stick_figures](../stick_figures/README.md) experiment predicts pose controls and passes them to a fixed renderer. That is text-to-pose, not learned pixel generation. Its clean drawings and angle-accuracy results are not evidence for this task.

## Train and generate

From the repository root with your existing PyTorch/NumPy/Pillow environment:

```powershell
python -m text2image.stick_pixels.prepare_data
python -m text2image.stick_pixels.train --steps 2000 --seed 7
python -m text2image.stick_pixels.evaluate
python -m text2image.stick_pixels.generate "torso deeply bent ; left arm raised ; right arm down ; left knee bent ; right knee straight"
```

Generation saves `samples/generated.png`. `--temperature 0` selects pixels greedily; the default 0.8 samples them. Use `--count`, `--seed`, and `--output` to control variation and destination. The vocabulary remains the restricted pose grammar; plain-English paraphrases are not implemented. Use `bent`, not `bended`.

Training starts fresh and selects `checkpoints/model.pt` by lowest validation pixel cross-entropy. There is no resume support yet. A new training run replaces this experiment's checkpoint and history; earlier angle-based/Fashion-MNIST/translation models are untouched.

## Synthetic images are targets, never an inference renderer

Offline preparation uses the existing stick-figure geometry to create labeled images. This is a synthetic dataset, like a program rendering training examples for a vision model. It is legitimate pixel supervision, but it does not mean the trained model itself knows that program.

Preparation converts drawings into **24 x 24 binary images**, with pixel ID 0 for background and 1 for ink. Every pixel is still a target predicted by the Transformer; all geometry remains confined to this offline generator.

### Version 2: many genuinely different drawings per pose

The default generator now creates:

| Split | Pose combinations | Images per combination | Total images |
|---|---:|---:|---:|
| Train | 84 | 256 | 21,504 |
| Validation | 12 | 64 | 768 |
| Test | 12 | 64 | 768 |

The initial version had only 24 records per combination, using at most nine one-pixel translations of a fixed drawing. Version 2 independently varies torso length, upper/lower arm and leg lengths, shoulder/hip width, head radius, neck length, elbow shape, stroke width, overall scale, and position. Small angle jitter is capped at three degrees, preserving the requested discrete pose state. There is no mirroring that would exchange left/right instructions. Full drawing bounds are fitted within the image before rasterization so limbs are not clipped.

The generator rejects identical binary rasters within or across splits. The measured default dataset contains **23,040 unique images**, and every pose has its requested unique variation count. Caption clause order also varies. Variant zero uses a deterministic centered reference style; the remaining variants randomize appearance. Differences smaller than a pixel can disappear at 24 x 24, which is why uniqueness is measured after rasterization.

Configure the count yourself:

```powershell
python -m text2image.stick_pixels.prepare_data --train-variants 256 --eval-variants 64 --seed 42
```

For example, `--train-variants 1024` creates 86,016 training images, still using the same held-out combination rules. Increasing training variants does not add training exposure to validation/test poses. Split-local random streams separate image sampling from caption-order sampling. The manifest records record counts, unique raster counts, generation seed, file checksums, and dataset fingerprint.

The original version-1 prepared files are archived under `data/translation_v1/`. The new data do **not** retroactively improve the existing checkpoint: fresh training is needed. New checkpoints record the data fingerprint; evaluation rejects a checkpoint/data mismatch. Generation can still run the old checkpoint and prints the dataset version it was trained on. Existing version-1 model/results were retained; no retraining was performed as part of generating version 2.

A review grid of eight variations of one fixed pose is at [samples/dataset-variations.png](samples/dataset-variations.png). Those are **offline training-target-style images**, not model outputs; the displayed pose belongs to the held-out test set and is never included in training.

The stored files contain only caption IDs, raster pixels, and combination IDs for split auditing. **The training loop loads only captions and pixels.** It does not train on or read combination IDs as model inputs, angles, joint coordinates, component classifiers, or renderer commands. Generation imports only the pixel model, vocabulary constants, PyTorch, and PNG-saving utilities.

All 108 defined combinations are still covered: 84 training, 12 validation, 12 test. Training excludes every combination pairing a deeply bent torso with a raised left arm. Validation reserves slightly bent torso with raised right arm. Each individual primitive state occurs in training. No held-out combination is rendered into training images, regardless of caption order or image translation. All variants for one combination stay in the same split.

This finite synthetic dataset and fixed image style are a first experiment. It is not a general cartoon-image dataset, and it does not establish broad language understanding.

## Actual model path

For a batch of `B` examples:

```text
Caption tokens, right-padded to 32 slots        [B,32]
Binary image targets, flattened row by row     [B,576]
Model input = caption + image[:, :-1]          [B,607]
Token and learned position embeddings          [B,607,64]
Three causal Transformer blocks                [B,607,64]
Final LayerNorm and linear pixel head           [B,607,2]
Keep score rows 31 through 606                  [B,576,2]
Cross-entropy against the 576 pixel targets     one scalar
```

There are 190,274 parameters: width 64, three pre-LayerNorm blocks, four attention heads, and feed-forward width 256. It uses no dropout in this initial version. The 32-slot caption includes a start/readout token, word tokens, separators, and padding. Pixel IDs 0 and 1 are distinct from caption PAD=2; caption IDs start above the pixel range.

The causal mask blocks future positions, preventing a supplied training pixel from leaking into its own prediction. The caption key-padding mask blocks unused caption slots. The final caption position still acts as a query and predicts pixel zero. Pixel zero is real background and contributes to loss; it is not replaced by `-100`. All 576 target pixels are graded.

During generation, the model starts with the caption only. It samples pixel 0, appends it, samples pixel 1, and repeats until it has 576 predicted binary values. Those values are reshaped into 24 rows of 24 pixels and saved directly to PNG. The only display operation is scaling to show pixels larger; it cannot repair an incorrect body or draw missing limbs.

Unlike the earlier angle model, this network must learn how instructions relate to complete raster structure. The output head has exactly **two categories**, not five pose controls. There is no postprocessing that enforces bones, joints, limb counts, or pose validity. Poor or noisy drawings are possible and must be reported as such.

## Optimization and evaluation

AdamW uses peak LR 0.0005, 100-step linear warmup, then absolute-step inverse-square-root decay; gradient norm is clipped to 1. Batch size defaults to 32. Validation teacher-forced loss is measured every 200 updates. Only validation selects checkpoints; test images are evaluated afterward.

Loss alone is especially misleading for sparse line drawings. An all-background image can have high pixel accuracy while drawing nothing. Evaluation therefore reports:

- Teacher-forced pixel cross-entropy on each split.
- **Foreground/ink intersection over union (IoU)** of autoregressively generated images against centered reference images.
- Number of completely blank generated images and mean generated ink count.
- A shuffled-caption alignment control: mismatch generated images with reference captions to test whether semantic pairing improves overlap.
- An all-background baseline, which has zero ink IoU for nonempty references.
- Expected/reference versus generated raster grids. Images in a reference column come from the dataset; generated columns contain only neural-network pixel predictions.

Free-running evaluation uses the manifest's variant stride to generate one centered-reference example per semantic combination, capped at 12 per split. Test includes all 12 held-out combinations. Exact pixel overlap is sensitive to one-pixel shifts and legitimate drawing variation, so inspect grids as well. Sampling is not guaranteed to recover the same translation as the centered reference; no alignment correction is applied. Neither good teacher-forced loss nor high pixel accuracy establishes correct pose composition.

The model and checkpoint are a baseline. No claim of successful held-out pixel composition is made without these generated-image results. In particular, the earlier angle model's 100% score cannot be transferred to this experiment.

## Files

| File | Role |
|---|---|
| [schema.py](schema.py) | Text and binary-pixel IDs, lengths, tokenization; no geometry |
| [prepare_data.py](prepare_data.py) | Offline raster training-data creation |
| [model.py](model.py) | Pixel Transformer and autoregressive sampling |
| [train.py](train.py) | Pixel-only cross-entropy, validation, best checkpoint |
| [generate.py](generate.py) | Text to directly predicted pixels to PNG |
| [evaluate.py](evaluate.py) | Generated-pixel overlap, blank checks, reference grids |
| [test_pixels.py](test_pixels.py) | Output categories, alignment, masks, gradients, data/split audits |

Local data, checkpoints, samples, and raw reports stay under this folder and are ignored by Git. See [RESULTS.md](RESULTS.md) for the measured initial run.

```powershell
python -m unittest -v text2image.stick_pixels.test_pixels
```
