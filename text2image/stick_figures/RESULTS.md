# Revised compositional stick-figure results

The original whole-caption pooled encoder fit the training combinations but failed the strict held-out pairing test in two of three seeds. Its implementation, checkpoints, raw metrics, and [baseline report](BASELINE_RESULTS.md) are preserved.

## What changed

The revised model encodes each semicolon-delimited clause **independently** with one shared Transformer. It learns a body-part identity distribution and a normalized angle from each clause, then softly routes the predicted angles into five output slots. Its training objective is pose MSE + clause-angle MSE + 0.25 times clause-identity cross-entropy. Clause-identity and angle targets come from the synthetic annotation schema **during training only**; inference only splits at semicolons and tokenizes words. There is no hard-coded phrase-to-angle lookup during generation.

This is an explicit factorization prior and additional structured supervision, not a claim that an unconstrained language model spontaneously learned disentanglement. The renderer and hand-designed five-part pose representation are unchanged. The revised model has **32,070 parameters**, versus 107,653 in the original.

## Same strict split, not a relaxed test

All 108 pose combinations remain present in the dataset. The same 84 train / 12 validation / 12 test split is used. No training caption combines a deeply bent torso and raised left arm, regardless of other limbs. Validation reserves slightly bent torso with raised right arm. Each individual body-part state occurs in training. Caption orders vary, but no semantic pose combination crosses splits.

We trained three predetermined seeds for 1,000 updates, initially with `--validation-only`. Those validation-selected checkpoints were frozen before running `--evaluate-only`; no further training followed the test results. The test split is already known from the earlier baseline experiment, so this is a developmental comparison, **not a fresh blind benchmark**. A future broader claim needs new held-out tasks.

| Seed | Selected update | Validation all five within 10 degrees | Test all five states correct | Test all five within 10 degrees |
|---|---:|---:|---:|---:|
| 7 | 300 | 100% | 100% | 100% |
| 11 | 800 | 100% | 100% | 100% |
| 23 | 900 | 100% | 100% | 100% |

Both validation and test each contain 288 caption records but only **12 distinct semantic poses**. These are small deterministic synthetic tasks, not hundreds of independent human poses.

| Seed | Torso MAE | Left arm MAE | Right arm MAE | Left knee MAE | Right knee MAE |
|---|---:|---:|---:|---:|---:|
| 7 | 1.82 deg | 2.06 deg | 1.15 deg | 1.60 deg | 1.24 deg |
| 11 | 2.51 deg | 6.01 deg | 3.94 deg | 1.22 deg | 1.61 deg |
| 23 | 2.04 deg | 4.03 deg | 1.66 deg | 1.02 deg | 0.87 deg |

Learned clause-to-part routing is 100% correct across the held-out test captions in all seeds. The no-text mean baseline has 0% joint state accuracy; shuffled-caption alignment has 6.25%. Those controls do not demonstrate unrestricted English understanding, but confirm that matching the varying instructions matters.

## Exhaustive caption-order audit

After freezing the checkpoints, we evaluated every one of the **120 possible clause orders** for each of the 12 test poses: 1,440 captions per seed, 4,320 total. This is a deterministic permutation audit, not 4,320 independent semantic tests.

| Seed | All five states correct | All five within 10 degrees | Worst individual angle error | Maximum prediction drift from reordering |
|---|---:|---:|---:|---:|
| 7 | 100% | 100% | 2.95 deg | 0.000057 deg |
| 11 | 100% | 100% | 7.89 deg | 0.000038 deg |
| 23 | 100% | 100% | 4.10 deg | 0.000038 deg |

Permutation invariance is encouraged structurally: clauses do not receive order-dependent clause-position embeddings; shared per-clause encoding plus normalized weighted aggregation is invariant to clause permutation except floating-point rounding. A unit test checks this, and also checks that changing one clause cannot change a different clause's local angle/identity predictions.

## Visual comparison

The same first held-out example is shown before and after. Blue is canvas-left; orange is canvas-right. Raw angle errors above are measured **before** clamping to the drawing range; rendering still clamps out-of-range angles as in the baseline.

- [Original seed-7 prediction](samples/example-seed-7.png)
- [Revised seed-7 prediction](samples/factorized/example-seed-7.png)
- [Revised six-pose grid](samples/factorized/held-out-seed-7.png)

## Reproduce

```powershell
python -m text2image.stick_figures.data
python -m text2image.stick_figures.train --architecture factorized --validation-only --steps 1000 --seeds 7 11 23
python -m text2image.stick_figures.train --architecture factorized --evaluate-only --seeds 7 11 23
python -m text2image.stick_figures.audit
```

Existing data need not be regenerated. Fresh training replaces that architecture's checkpoints. Original artifacts remain at their old paths; revised artifacts use `factorized/` under checkpoints, samples, and results. Raw reports are `results/factorized/validation.json`, `metrics.json`, and `permutation_audit.json`.

## What is and is not fixed

The revised model reliably passes this finite held-out composition test across the tested seeds and all clause orders. It has not learned arbitrary pose language, body topology, 3D bending, appearance preservation, or interacting constraints such as touching both hands together. Successful composition here relies on known independent control slots and clause supervision. Expanding the language interface is a separate next step, with its own language-generalization tests.
