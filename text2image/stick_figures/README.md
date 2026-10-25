# Stick figures: testing unseen combinations of pose instructions

This experiment asks a specific question: **can a small Transformer combine a familiar torso instruction and a familiar arm instruction when that pairing never occurs in training?** It generates a new pose from text. It does not edit an input photograph or preserve a person's identity.

## Measured first result

See [RESULTS.md](RESULTS.md) for the three-seed report. After 2,000 updates, seed 7 achieves 62.5% held-out all-five nearest-state accuracy; seeds 11 and 23 achieve 0%. All three achieve 0% when every angle must be within ten degrees. Training accuracy is approximately 100%. **This baseline does not yet demonstrate reliable compositional generalization.** Arm control is the principal failure, despite successful fitting of familiar combinations.

## Run from the repository root

Use the existing environment with `torch`, `numpy`, and `Pillow`:

```powershell
python -m text2image.stick_figures.data
python -m text2image.stick_figures.train --steps 2000 --seeds 7 11 23
python -m unittest -v text2image.stick_figures.test_experiment
```

These commands generate the local dataset, train three independently initialized models, select each checkpoint using validation only, and evaluate the test set after selection. A fresh run overwrites this experiment's checkpoints, metrics, and previews. Resume is not implemented. Existing Fashion-MNIST and translator artifacts are unaffected.

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

## What the model learns, and what is supplied by us

```text
Caption words
  -> integer IDs and learned word/position embeddings
  -> two bidirectional Transformer encoder blocks
  -> representation at a learned start/readout token
  -> a shared feed-forward head producing five continuous pose controls
  -> fixed skeleton geometry
  -> PNG stick figure
```

The model does not predict a single whole-pose category or retrieve a drawing. It predicts five continuous values, supervised independently by known pose controls. All five predictions share the same learned text representation and output network. It has no hard-coded rule saying that the phrase `left arm raised` should set output 1 to a particular number; learning that language-to-control mapping is the task.

However, **the output representation and skeleton geometry are strong, hand-designed structural priors**. We supply a five-part pose representation, valid limb connectivity, and how angle controls move those limbs. This tests compositional language-to-control learning; it does not show that the model discovered anatomy, learned to draw raw pixels, or independently invented body-part representations.

### One input and forward pass

```text
torso deeply bent ; left arm raised ; right arm down ; left knee straight ; right knee straight
```

Clause ordering varies in training. The vocabulary contains the words used by these clauses and a semicolon separator. The tokenizer only lowercases, splits words/separators, and looks up IDs; it does not interpret the instructions. It prepends a readout token and pads to 32 slots. A batch therefore has:

```text
input IDs                  [batch,32]
word + position vectors    [batch,32,64]
encoder block 1            [batch,32,64]
encoder block 2            [batch,32,64]
readout at position 0      [batch,64]
LayerNorm + Linear/GELU/Linear head
predicted controls         [batch,5]
```

Each block has width 64, four attention heads, feed-forward width 256, pre-layer normalization, and dropout 0.1. Padding keys are masked; attention is bidirectional because the complete instruction is known. There is no causal generation loop or pixel-token output sequence.

Targets are normalized angles: divide torso by 50, arm angles by 160, and knee angles by 65. The loss is mean squared error over the five normalized controls and all batch examples. A `raised` arm is target 1, an `out` arm is 0.5, and a `down` arm is 0. The shared output head has no sigmoid; raw predictions may go slightly outside [0,1]. Evaluation uses those raw predictions, while rendering clamps to [0,1] to keep drawings on canvas.

Training uses batch size 64, AdamW with weight decay 0.01, gradient-norm clipping at 1, and an absolute-step LR warmup to 0.001 over 100 updates followed by inverse-square-root decay. Every 100 updates, validation MSE selects the best checkpoint. Seeds 7, 11, and 23 are predetermined, not selected for test performance.

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

Generation prints raw predicted angles and saves `samples/generated.png`. The model is deterministic for a given prompt and checkpoint; it does not currently sample multiple people or styles.

## Files and outputs

| File/path | Role |
|---|---|
| [data.py](data.py) | Enumerate all combinations, isolate split pairings, produce captions and targets |
| [model.py](model.py) | Small text encoder and continuous-control prediction head |
| [render.py](render.py) | Fixed skeleton, colors, and comparison grids |
| [train.py](train.py) | Train seeds, select by validation, evaluate once after selection, save metrics |
| [generate.py](generate.py) | Prompt-to-pose-to-PNG command |
| [test_experiment.py](test_experiment.py) | Exhaustive coverage, leakage, primitive coverage, skeleton, and gradient checks |
| `checkpoints/seed-N.pt` | Best-validation checkpoint for each seed |
| `results/metrics.json` | Measured metrics, loss history, and selected preview predictions |
| `samples/held-out-seed-N.png` | Expected vs predicted figures for a fixed six test combinations |

Preview combinations are fixed before observing predictions, not cherry-picked successes. The expected figure is a reference from the generator, not another model output. Both reference and prediction use the same deterministic renderer.
