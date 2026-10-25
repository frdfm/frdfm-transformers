# First compositional stick-figure experiment

Complete 108-combination universe; 84 training, 12 validation, 12 test combinations. Each combination has 24 shuffled-clause captions. Training excludes **all** cases pairing deeply bent torso and raised left arm; those are test-only. Validation reserves slightly bent torso with raised right arm. Every individual component state occurs in training.

Three predetermined seeds; 2,000 updates each; 107,653 parameters; GPU RTX 3060. Checkpoint chosen by validation MSE. The table uses nearest-state accuracy, which allows sizeable angle errors; strict geometric accuracy is reported separately.

| Seed | Selected update | Train all-five states | Validation all-five states | Test all-five states | Test all-five within 10 degrees |
|---|---:|---:|---:|---:|---:|
| 7 | 2000 | 100.0% | 100.0% | 62.5% | 0.0% |
| 11 | 2000 | 99.9% | 22.6% | 0.0% | 0.0% |
| 23 | 1800 | 100.0% | 11.5% | 0.0% | 0.0% |

Test accuracy is over 288 caption records, representing **12 semantic pose combinations**, not 288 independent poses. Multiple clause orderings test prompt-order sensitivity. These are descriptive results without statistical confidence claims.

| Seed | Torso MAE (degrees) | Left arm MAE | Right arm MAE | Left knee MAE | Right knee MAE |
|---|---:|---:|---:|---:|---:|
| 7 | 3.1 | 38.3 | 26.8 | 2.6 | 2.5 |
| 11 | 1.9 | 67.6 | 63.8 | 1.9 | 2.4 |
| 23 | 2.9 | 56.9 | 56.2 | 2.2 | 2.9 |

The no-text training-mean baseline scores 0% all-five state accuracy for all three runs. Shuffling predictions against test captions scores 3.8% (seed 7), 0.0% (seed 11), 0.7% (seed 23).

**Interpretation:** strong fitting of training combinations does not establish composition. Seed 7 partially transfers the withheld pairing, but the arm angles are inaccurate. Seeds 11 and 23 fail the joint test. All three have zero strict all-five geometric success. The generic pooled text encoder has not learned reliable independent arm control in this setting.

The system predicts five hand-designed pose controls and uses fixed skeleton geometry. It is not learning body topology or raster drawing, and this small grammar does not establish unrestricted language understanding. Blue limbs are canvas-left; orange limbs are canvas-right. The torso is tilted in a 2D plane, not physically bent in 3D.

Predetermined seed-7 example: expected `[50,160,0,0,0]` degrees; predicted [47.1, 134.6, 14.1, -1.0, 2.2]. The first example and first six grid combinations were fixed before looking at predictions.

[Seed 7 first example](samples/example-seed-7.png) | [Seed 7 comparison grid](samples/held-out-seed-7.png) | [Seed 11 grid](samples/held-out-seed-11.png) | [Seed 23 grid](samples/held-out-seed-23.png)

Raw local results: `results/metrics.json`, with all split metrics, validation histories, and preview angles. Generated data/checkpoints/images/raw metrics are ignored by Git; this summary preserves the measured outcome.

Reproduce with `python -m text2image.stick_figures.data` then `python -m text2image.stick_figures.train --architecture pooled --steps 2000 --seeds 7 11 23`. Re-running replaces the local artifacts. GPU training need not be bitwise deterministic across runs.

An earlier exploratory run withheld complete five-part tuples while still showing their constituent pairings during training. It was not a sufficiently strict test of the motivating bend-and-raised-arm request; those outcomes are excluded from this report. The results above use fresh training on the stronger pairing-held-out split.
