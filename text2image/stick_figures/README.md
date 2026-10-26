# Text-to-pose controls: testing unseen instruction combinations

**This is not pixel generation.** The network predicts five angles; a fixed renderer draws the skeleton. Its clean images must not be presented as learned text-to-image results. The direct pixel experiment is [stick_pixels](../stick_pixels/README.md).

This experiment asks a specific question: **can a small Transformer combine a familiar torso instruction and a familiar arm instruction when that pairing never occurs in training?** It generates a new pose from text. It does not edit an input photograph or preserve a person's identity.

## Current result and commands

The revised **factorized** model now passes the reserved deep-torso-bend + raised-left-arm combination test across all three tested seeds, including the stricter requirement that all five angles are within 10 degrees. All 120 clause orderings for every test pose also pass. See [RESULTS.md](RESULTS.md) for measured errors, baselines, and caveats. The [original failure report](BASELINE_RESULTS.md) is preserved.

Run from the repository root using the existing `torch`, `numpy`, and `Pillow` environment:

```powershell
python -m text2image.stick_figures.data
python -m text2image.stick_figures.train --architecture factorized --validation-only --steps 1000 --seeds 7 11 23
python -m text2image.stick_figures.train --architecture factorized --evaluate-only --seeds 7 11 23
python -m text2image.stick_figures.audit
python -m unittest -v text2image.stick_figures.test_experiment
```

The first training command selects models using validation without evaluating or drawing test poses. The second only loads those frozen checkpoints and evaluates test data. The audit checks all 120 clause orders per test pose, without training. Ordinary `train --architecture factorized --steps 1000` trains and then evaluates automatically. Default architecture is `factorized`; default update count remains 2,000 if omitted. The measured revised experiment used 1,000.

Resume is not implemented. Fresh training overwrites only the requested architecture's checkpoints and reports. Original pooled-model artifacts retain their old paths; revised artifacts use a `factorized/` subfolder. Fashion-MNIST and translation artifacts are unaffected.

## The complete combinatorial dataset

Every example specifies five components:

| Component | Allowed states | Target angles in degrees |
|---|---|---|
| Torso | upright / slightly bent / deeply bent | 0 / 25 / 50 |
| Left arm | down / out / raised | 0 / 80 / 160 |
| Right arm | down / out / raised | 0 / 80 / 160 |
| Left knee | straight / bent | 0 / 65 |
| Right knee | straight / bent | 0 / 65 |

The complete universe contains `3 * 3 * 3 * 2 * 2 = 108` combinations. The generator enumerates all of them, splits by semantic constraints, **then** creates 24 shuffled-clause captions per combination. Some clause orderings can repeat within a split; no semantic combination crosses splits.

| Split | Rule | Combinations | Caption records |
|---|---|---:|---:|
| Train | Everything not reserved below | 84 | 2,016 |
| Validation | Torso slightly bent **and** right arm raised; all other components vary | 12 | 288 |
| Test | Torso deeply bent **and** left arm raised; all other components vary | 12 | 288 |

The validation and test rules cannot overlap because the torso states differ. Every primitive state remains in training: deep bends occur with other left-arm states, and raised left arms occur with other torso states. But **not a single training example pairs a deep bend with a raised left arm**, even with different legs or a different right arm. This is stronger than withholding just one complete five-component tuple.

The validation set reserves a different unfamiliar pairing for checkpoint selection. The test set is not used for early stopping or choosing a checkpoint. The split is a defined challenge, not a claim that a random subset of all possible pose semantics has been covered.

Dataset output is `data/dataset.json`: every combination ID, its split, caption text, and five numerical targets are inspectable. No downloads, human annotations, or pretrained model are required.

## Revised model: separate instruction meaning before composing the pose

The original model compressed every instruction into one shared vector and predicted all five controls from it. It often fit familiar combinations but confused arm controls in unfamiliar combinations. The revised architecture supplies a stronger independence bias:

```text
Five clauses in any order
 -> split at semicolons, tokenize words (syntax only)
 -> shared Transformer encodes each clause separately
 -> each clause predicts its body-part identity and normalized angle
 -> soft learned routing combines angle predictions into five control slots
 -> fixed skeleton renderer draws the controls
```

There is no rule at inference mapping the phrase `left arm raised` directly to 160 degrees. The model learns word embeddings, body-part identification, and angle prediction from annotations. However, the five output slots, clause boundaries, independent-clause computation, and skeleton geometry are **hand-designed structure**. This experiment tests composition under that structure; it does not show that the model independently invented a body representation.

### A batch through the revised model

For example:

```text
torso deeply bent ; left arm raised ; right arm down ; left knee straight ; right knee straight
```

`encode_clauses` splits only on semicolons. It requires five clauses, tokenizes each, prepends a readout token, and pads each clause to six positions. It neither identifies the part nor interprets the state. With batch size `B`:

```text
clause IDs                           [B,5,6]
flatten for shared text encoder      [B*5,6]
word + local position embeddings     [B*5,6,48]
one Transformer encoder block        [B*5,6,48]
readout vector and LayerNorm         [B*5,48]
restore clause dimension             [B,5,48]
learned part logits                  [B,5,5]
learned local angle                  [B,5]
```

Each clause uses one shared 48-wide Transformer block with four attention heads, feed-forward width 192, pre-LayerNorm, and dropout 0.1. Padding keys are masked. Attention is bidirectional *within* a clause; there is no attention across clauses. All clauses share parameters, but receive only local position embeddings, so reordering clauses does not change their individual meaning vectors.

The part head predicts five logits per clause; the angle head predicts one continuous value. At training time, `clause_supervision` supplies the known part identity and local angle for each synthetic clause. It is never called by generation. This is additional structured supervision beyond the original whole-pose MSE target.

### Exact learned routing and loss

Let `part_logits[b,c,k]` score body part `k` for clause `c`. Routing uses:

```python
routes = part_logits.softmax(dim=-1)  # probability per body part for each clause
routes = routes / routes.sum(dim=1, keepdim=True).clamp_min(1e-8)
controls = (routes * clause_angles.unsqueeze(-1)).sum(dim=1)
```

The second normalization makes weights sum to one across clauses for each output part. If a clause is confidently identified as left arm, its angle dominates the left-arm output. This routing is learned, not a text-to-slot dictionary in generation.

Training minimizes:

```text
MSE(final five controls, expected five controls)
+ MSE(local clause angles, expected local angles)
+ 0.25 * cross_entropy(part logits, expected part identities)
```

Angles are normalized by their component's maximum: 50 degrees for torso, 160 for arms, and 65 for knees. The learned angle head is not constrained by a sigmoid. Evaluation uses raw predictions; rendering clamps to [0,1].

The model has **32,070 parameters**. The unchanged skeleton connects those five predicted controls into a drawing. There is no pretrained text encoder, pose retrieval, full-pose category lookup, or per-word rule in generation. Clause-order invariance is supplied by the architecture, not merely hoped for from shuffled training examples.

### Training and checkpoint selection

Both architectures use batch size 64, AdamW with weight decay 0.01, gradient-norm clipping at 1, and LR warmup to 0.001 over 100 absolute updates followed by inverse-square-root decay. Validation MSE selects a checkpoint every 100 updates. The selected revised checkpoints came from updates 300, 800, and 900 for seeds 7, 11, and 23 in the measured 1,000-update run.

The **pooled** baseline remains available through `--architecture pooled`. It has a two-block, width-64 whole-caption encoder and a shared five-control head, trained only with whole-pose MSE. `train --architecture pooled --steps 2000 --seeds 7 11 23` reproduces its original recipe. Generation selects the matching model class from checkpoint metadata and rejects an architecture mismatch.

### Drawing conventions

The renderer never sees a caption. It receives only the five predicted numbers and connects a fixed skeleton. Arms are blue for left and orange for right. **Left/right mean canvas-left/canvas-right**, not anatomical left from an arbitrary camera view. Torso bending is a 2D tilt toward canvas-right, a simplified depiction of bending forward; it is not a physically accurate 3D body or balance simulation. Raised arms point upward in canvas coordinates, independent of torso tilt. Bent knees rotate the lower leg outward.

All variants share a fixed body size and style. This first experiment varies instructions and their combinations, not appearance, body proportions, camera, or artistic style. Predicting numerical controls lets us score instruction adherence directly rather than judge image similarity alone.

## Evaluation: what would count as success?

The script writes `results/metrics.json` with training, validation, and held-out-test results for every seed:

- **Per-component state accuracy:** assign each predicted angle to its nearest defined state. Did each requested component land in the correct region?
- **All-five state accuracy:** all five components must be correct in the same example. This is stricter than averaging component accuracies, but broad state intervals can conceal sizeable angle errors.
- **Mean absolute angle error:** reported separately for torso, arms, and knees, in degrees.
- **All-five within 10 degrees:** every component must be within ten degrees of its exact target, a stricter geometric criterion.
- **No-text baseline:** always predict the training-set mean controls.
- **Shuffled-caption control:** permute the model's test predictions relative to their expected poses. This keeps the same output distribution but breaks alignment with the instructions, testing whether matching the text matters. Since all test captions share the reserved torso/left-arm pairing, only the other components vary in this control.

A good result should beat both controls, achieve low angle errors, and be reasonably consistent across seeds. One successful seed is insufficient evidence of reliability. It also does not prove an internally disentangled representation, unrestricted English understanding, or generalization outside this 108-combination world.

The test captions use the same vocabulary and clause grammar as training. This is a held-out **meaning-combination** test, not a test of unseen synonyms, natural conversation, or language paraphrases.

## Try your own supported combination

After training:

```powershell
python -m text2image.stick_figures.generate "torso deeply bent ; left arm raised ; right arm down ; left knee straight ; right knee straight"
```

The default loads seed 7; `--seed 11` or `--seed 23` loads those runs. Use all five clauses, with one of the defined states per component. Unknown words are rejected. The tokenizer accepts known-word inputs even if they are contradictory or incomplete, but those are outside the demonstrated training task.

Generation prints raw predicted angles and saves `samples/factorized/generated.png` by default. Use `--architecture pooled` to load the original model and its legacy paths. The model is deterministic for a given prompt and checkpoint; it does not currently sample multiple people or styles.

## Files and outputs

| File/path | Role |
|---|---|
| [data.py](data.py) | Enumerate all combinations, isolate split pairings, produce captions and targets |
| [model.py](model.py) | Small text encoder and continuous-control prediction head |
| [render.py](render.py) | Fixed skeleton, colors, and comparison grids |
| [train.py](train.py) | Train seeds, select by validation, evaluate once after selection, save metrics |
| [generate.py](generate.py) | Prompt-to-pose-to-PNG command |
| [audit.py](audit.py) | Exhaustive clause-order testing of frozen revised checkpoints |
| [test_experiment.py](test_experiment.py) | Exhaustive coverage, leakage, primitive coverage, skeleton, and gradient checks |
| `checkpoints/factorized/seed-N.pt` | Revised best-validation checkpoint for each seed; original files remain one directory above |
| `results/factorized/metrics.json` | Revised measured metrics, loss history, and selected preview predictions |
| `samples/factorized/held-out-seed-N.png` | Revised expected vs predicted figures for six fixed test combinations |

Preview combinations are fixed before observing predictions, not cherry-picked successes. The expected figure is a reference from the generator, not another model output. Both reference and prediction use the same deterministic renderer.
