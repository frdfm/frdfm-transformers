# First direct-pixel stick-figure result

**Historical run: dataset version 1 (24 records per combination, translation-only variation).** The current prepared data are version 2, with 256 training and 64 evaluation drawings per pose and randomized body/style variation. Version 2 has not been retrained or evaluated yet. The metrics below remain the original version-1 results; do not attribute them to the new dataset. Original data are archived at `data/translation_v1/`.

## What was actually trained

A 190,274-parameter autoregressive Transformer predicts 576 binary pixels from text and preceding pixels. Its output head has two categories (background/ink). No angle, joint, pose-control, body-part routing, or skeleton-validity target is used in optimization. Generation does not import or invoke a skeleton renderer. The PNG is a direct display of neural-network pixel predictions.

An offline renderer creates synthetic raster training images. That use is restricted to dataset preparation. The training loop loads only caption IDs and raster pixels; combination metadata is used for split auditing, not prediction.

## Setup

- All 108 semantic pose combinations are covered: 84 train, 12 validation, 12 test.
- Every deep-bend + raised-left-arm combination is test-only. A different pairing is validation-only. All individual component states occur in training.
- 24 image/caption variants per combination. Variant zero is centered; others translate by at most one pixel. Clauses are shuffled.
- One seed (7), 2,000 updates, batch size 32, RTX 3060. About 131 seconds; peak allocated GPU memory 0.93 GiB.
- Validation selected update 1,400. No test image selected the checkpoint.
- Generation evaluates one centered-caption example for each test combination (12), and a fixed first 12 combinations from train/validation. These tiny subsets are not a broad benchmark.

## Measured output

Foreground IoU is intersection divided by union of ink pixels. It is more informative here than accuracy over mostly empty background.

| Split | Teacher-forced loss | Greedy ink IoU | Shuffled-caption alignment ink IoU | Blank images |
|---|---:|---:|---:|---:|
| train | 0.0381 | 0.354 | 0.374 | 0 / 12 |
| val | 0.0600 | 0.160 | 0.177 | 0 / 12 |
| test | 0.1164 | 0.480 | 0.487 | 0 / 12 |

At temperature 0.8, test ink IoU is 0.343. All-background predictions have zero ink IoU, but about 95% pixel accuracy. This demonstrates why background-dominated accuracy and low teacher-forced loss do not establish instruction following.

**Conclusion:** the actual pixel model draws nonblank, stick-like structures, but this run does **not** demonstrate reliable pose composition or prompt control. Shuffling generated images against captions does not reduce overlap. Some plausible-looking generated shapes can therefore reflect common image structure rather than satisfaction of the requested components. We must not replace this conclusion with the earlier angle model's 100% pose-control scores.

The evaluation is sensitive to pixel shifts and uses only one generation per reference pose for these metrics. No pose classifier, joint extractor, alignment correction, or geometry repair is applied to improve the apparent outputs. The sampled result is a separate fixed-seed generation, not a selected best sample.

## See the actual rasters

- [First held-out reference/prediction](samples/test-example.png): reference is first, neural-network pixels second.
- [All twelve held-out comparisons](samples/test-comparison.png).
- [Sampled test outputs](samples/test-sampled.png).
- [Training comparisons](samples/train-comparison.png).

The first example is predetermined, not cherry-picked. Enlarging the 24 x 24 bitmap does not add detail. Actual reference/prediction arrays are saved as `results/{train,val,test}-generated-pixels.npz`; full metrics are in `results/evaluation.json`, and validation history is in `results/history.json`.

## Original recipe and current data

```powershell
python -m text2image.stick_pixels.prepare_data
python -m text2image.stick_pixels.train --steps 2000 --seed 7
python -m text2image.stick_pixels.evaluate
```

Running preparation now creates version 2, not the original version-1 dataset described above. Its training/evaluation results will be a new experiment. The original checkpoint still exists locally, so generation can be run without preparing or training again. A new training invocation starts fresh and overwrites the pixel checkpoint; it does not touch other experiments. Resume is not implemented.
