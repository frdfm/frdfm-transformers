# Text2Stick: plain English to actual image pixels

Describe a stick figure in ordinary English. A text encoder and an autoregressive image decoder generate a **24 x 24 black-and-white bitmap**, one pixel at a time. The model does not output angles, joints, body-part controls, or drawing commands. There is no pose parser or skeleton renderer in generation.

This is a small model trained locally from scratch, not a pretrained general-purpose image generator. It learns from synthetic English descriptions of a limited pose world: upright/slightly/deeply bent torso, arms down/out/raised, and knees straight/bent. You can use ordinary sentences rather than the older semicolon syntax. It can tokenize unfamiliar words, but that does not guarantee it understands them.

## Use the trained version

Run from the repository root in your existing venv:

```powershell
python -m text2image.text2stick.generate "Draw a person bending forward and raising their left hand."
python -m text2image.text2stick.generate "Draw someone standing upright with both hands raised above their head." --count 4 --seed 123
python -m text2image.text2stick.generate "A person with their right arm stretched out and their left knee bent." --temperature 0.7
```

The included checkpoint is ready to try, but image quality and instruction following are still limited; see [measured results](RESULTS.md) and the fixed demo. Output: `text2image/text2stick/samples/generated.png`. Change `--output` to retain multiple grids. `--temperature 0` uses greedy pixels, while positive temperatures sample. `--guidance 1` disables classifier-free guidance; default 2 strengthens the text condition relative to the learned unconditional image distribution. Different seeds change samples. The guidance is entirely neural: it combines conditional/unconditional pixel logits, not inferred pose controls.

You need not specify all body parts. Training includes partial descriptions; unspecified details are sampled from what the model learned. They are not assigned an angle by a rules engine. This makes multiple drawings for one prompt legitimate. Outputs can still be malformed or fail to follow an instruction.

The model has not been taught arbitrary actions, objects, clothing, facial expressions, scene composition, or realistic 3D anatomy. "Waving" can be encoded but has no guaranteed learned action meaning; try explicit descriptions such as "raising the left hand". Left/right follow the synthetic dataset's **canvas-left/canvas-right** convention, not an anatomical camera transform. Forward bending is a stylized 2D torso tilt, not a realistic 3D motion.

## Where the data come from

The source is the varied binary-raster dataset in [stick_pixels](../stick_pixels/README.md). Its geometry is used only in offline dataset preparation. This experiment pairs those actual raster targets with natural descriptions, then trains directly on the pixels. No angle, joint, or part-identity target appears in the model loss.

There are 108 defined combinations of five pose components. The source reserves a full pairing, not just a single complete pose:

- Training never pairs deeply bent torso with raised left arm.
- Validation reserves slightly bent torso with raised right arm.
- Test reserves deeply bent torso with raised left arm, with every other limb combination.
- Each individual component state occurs in training.

The new language/image splits are:

| Split | Images/captions | What is held out |
|---|---:|---|
| `train` | 18,816 | First 224 style variants of each of 84 training poses |
| `word_val` | 1,344 | 16 fresh styles per familiar pose, using validation-only phrase family |
| `word_test` | 1,344 | Another 16 fresh styles per familiar pose, using test-only phrase family |
| `pose_val` | 768 | All 12 validation pose combinations, familiar wording families |
| `pose_test` | 768 | All 12 test pose combinations, familiar wording families |
| `combined_test` | 768 | Test poses plus test-only wording family |

`pose_test` and `combined_test` intentionally reuse reference images but change descriptions; these are not independent image sets. No training raster or complete caption is duplicated into a held-out split. Every variation of a reserved pose remains outside training. The test sets are not used to choose training checkpoints.

Training uses several wording families and random clause order, referring to arms as arms or hands and using expressions such as lifting, raising, overhead, down, sideways, leaning, and bending. Roughly 30% of training captions specify only one to three components. Their images still contain full figures, so the network must model unspecified details rather than fill them using a fixed parser. Some generated captions have synthetic phrasing; this is not a human-written language benchmark.

Wording validation/test use separately defined phrase/sentence families. These are meaningful held-out wording tests inside a small controlled vocabulary, not evidence of general English understanding. The exact captions are inspectable in `data/*-captions.json`.

## What enters the model

A byte-level byte-pair-encoding tokenizer is trained on **training captions only**. Its full byte alphabet can represent unfamiliar text without an unknown-word rejection. It adds a text end marker and padding, up to 160 tokens; overly long descriptions are rejected explicitly rather than silently truncated. Encoding is token lookup, not interpretation of pose instructions.

Text and image inputs use separate vocabularies:

```text
English text IDs: text tokenizer vocabulary; PAD=0
Pixel input IDs: 0=background, 1=ink, 2=begin-image
Pixel output categories: 0=background, 1=ink only
```

The image contains 576 pixels. During training:

```text
English token IDs                                 [batch, up to 160]
trim trailing padding unused by the whole batch
word + learned text-position embeddings           [batch, text_length, 96]
two bidirectional text-encoder blocks             [batch, text_length, 96]
English memory                                    [batch, text_length, 96]

image decoder input = begin-image + pixels[:-1]   [batch, 576]
pixel + learned image-position embeddings         [batch, 576, 96]
four image-decoder blocks                         [batch, 576, 96]
final LayerNorm + linear output head              [batch, 576, 2]
compare against all original 576 pixel targets
```

Each image block performs:

```text
LayerNorm -> causal pixel self-attention -> residual addition
LayerNorm -> cross-attention to English memory -> residual addition
LayerNorm -> Linear(96,384) -> GELU -> Linear(384,96) -> residual addition
```

There are four heads, each of width 24. The text encoder also uses pre-LayerNorm blocks but no causal mask: the complete description is known. Each block has independent parameters. The default network has **937,250 parameters**.

The English padding mask is applied in every encoder attention module AND every decoder cross-attention module. Pixel causality blocks future image inputs. Input begin-image predicts pixel 0; each supplied pixel then predicts the next. Every pixel, including background zero, is a valid target. There is no image padding or `-100` loss target.

## Training makes use of text, not just prior pixels

Sparse images encourage autoregressive models to rely on neighboring pixels while ignoring text. This version adds two explicit training mechanisms:

1. **Mismatched-caption ranking.** The same training image is scored with its actual caption and with a full caption from another training pose. A hinge penalty encourages lower image negative log-likelihood under the correct description by margin 0.025. It uses only text from training pose combinations; no held-out pairing is introduced as a negative description. Ranking applies only when the positive description fully specifies the pose, since a partial positive could legitimately match other full descriptions.
2. **Condition dropout.** Ten percent of positive conditions are replaced with the text end marker alone. This trains an unconditional pixel distribution for classifier-free guidance during sampling.

The main pixel loss is cross-entropy with ink pixels weighted twice as much as background, normalized by the total weight per image. The ranking term is unweighted per-pixel cross-entropy. Total objective:

```text
weighted positive pixel cross-entropy
+ 0.5 * mean_valid(max(0, 0.025 + positive_NLL - mismatched_NLL))
```

This is still pixel-only supervision; the loss never compares angles or joints. Mismatched descriptions use the data's combination metadata only to ensure a different training pose is selected. The generator at inference cannot access those IDs or labels.

AdamW uses weight decay 0.01, peak LR 0.0005, 200-update warmup, then absolute-step inverse-square-root decay. Batch size defaults to 32. Gradients are clipped to norm 1. Supported CUDA hardware uses BF16 autocast; otherwise training uses FP32. Validation selects by the equally weighted mean of pose-validation and wording-validation pixel NLL on fixed stratified style subsets. Test images and prompts do not select checkpoints.

## Sampling is actual pixel generation

The text is encoded once. Image generation starts with the begin-image token, then samples 576 binary pixels. Self-attention keys/values are cached as the prefix grows; cross-attention projections of English memory are cached too. A regression test checks cached outputs against recomputing the complete prefix. The cache accelerates the same learned computation; it does not draw a skeleton or enforce body geometry.

The final sequence is reshaped to 24 x 24 and enlarged with nearest-neighbor scaling for display. There is no stroke correction, image retrieval, pose-validity repair, angle predictor, or post-generation renderer. A malformed bitmap remains malformed.

## Reproduce, resume, and evaluate

Existing source data must first be prepared using the version-2 stick-pixel generator. Then:

```powershell
python -m text2image.text2stick.prepare_data
python -m text2image.text2stick.train --steps 4000
python -m text2image.text2stick.evaluate --temperature 0.8
python -m text2image.text2stick.demo
```

Data preparation learns a tokenizer and replaces this experiment's prepared files. Fresh training replaces this experiment's checkpoints and history. All earlier models are kept in their own directories.

Continue the current training trajectory with:

```powershell
python -m text2image.text2stick.train --resume --steps 2000
```

Resume restores weights, optimizer, absolute update, best validation value, and saved random state. It loads `checkpoints/latest.pt`; generation loads the validation-selected `checkpoints/model.pt`. These can represent different updates. A data-fingerprint mismatch is rejected on resume; generation checks tokenizer identity.

Evaluate validation only, without touching final test sets:

```powershell
python -m text2image.text2stick.evaluate --splits pose_val word_val --count 12
```

The default evaluation reports unfamiliar wording, unfamiliar pose combinations, and both together. It measures teacher-forced NLL, NLL increase for a mismatched caption, foreground intersection-over-union of generated images against actual reference images, blank outputs, and shuffled-caption image alignment. References are particular random styles, so exact pixel overlap is sensitive to legitimate style variation. Compare the reference/generated grids and try your own prompts; loss alone is insufficient.

The demo uses a fixed set of ordinary-English prompts, including partial instructions and left/right changes. It does not select or repair generated images. The user remains the judge of visual usefulness.

## Optional masked-pixel experiment

`--mode masked` selects a separate experimental checkpoint. This branch starts from all masked pixels and fills the bitmap over 16 confidence-based rounds with bidirectional image attention. Training supervises masked pixels only. It still predicts pixels, with no pose controls. Its checkpoint and results live under `checkpoints/masked/` and `results/masked/`.

The local trial initialized this branch from the locally trained autoregressive checkpoint. Its greedy outputs often became almost blank, so **autoregressive remains the default**. The masked branch is retained for inspection, not recommended as an improvement. Its fully masked cross-entropy is not directly comparable to autoregressive teacher-forced loss.

```powershell
python -m text2image.text2stick.generate "Draw someone with both hands raised." --mode masked
```

## Artifacts and tests

| Path | Meaning |
|---|---|
| `data/tokenizer.json` | Locally trained text tokenizer |
| `data/*.npz`, `*-captions.json` | Pixel targets and inspectable English captions |
| `data/manifest.json` | Split counts and data/tokenizer identities |
| `checkpoints/latest.pt` | Latest checkpoint for resume |
| `checkpoints/model.pt` | Best-validation checkpoint for generation |
| `results/history.json` | Training and validation measurements |
| `results/evaluation-t0.8.json` | Held-out generated-image measurements |
| `samples/generated.png` | Your most recent generation |
| `samples/autoregressive/plain-english-demo.png` | Fixed ordinary-English prompt grid |

```powershell
python -m unittest -v text2image.text2stick.test_model
```

Tests verify cache equivalence, pixel causality, both English masking applications through padding invariance, gradient flow to text and pixels, dataset isolation, and tokenization of ordinary sentences. Measured run details belong in [RESULTS.md](RESULTS.md); do not substitute the old angle-model's accuracy for this model's image quality.
