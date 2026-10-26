# Clean 48-pixel Text2Stick trial

The synthetic training targets are corrected: connected shoulders, visible head outlines, and strokes that survive downsampling. The model generates actual image pixels, but following English pose instructions is still unreliable.

## Data and model

- 18,816 training image/caption pairs: 84 combinations with 224 style variants each.
- All prepared targets pass eight-connected ink checks. Independent splits contain no identical rasters. Held-out pose pairings remain excluded from training.
- Targets are 48 x 48 binary images drawn by the offline renderer. Generation never imports that renderer or repairs its output.
- Each 2 x 2 pixel patch is encoded losslessly as a four-bit integer, giving 576 image tokens and 16 output categories. This supports every possible binary image, not just stick figures or valid poses.
- 939,952 parameters, two text encoder blocks, four image decoder blocks, width 96, four attention heads.

## Final training run

Trained from scratch for 4,000 optimizer updates with batch size 32 and seed 7. The final run took about 267 seconds on the local CUDA GPU, with peak allocated GPU memory 0.79 GiB. Preparation, evaluation, and earlier trials are excluded from this timing; allocated memory is not total GPU reservation.

Validation selected update 4000, with average per-pixel NLL 0.05913 on fixed stratified wording/pose validation subsets. In this run, best and latest checkpoints both happen to be update 4,000. Generation uses `checkpoints/model.pt`; resume uses `checkpoints/latest.pt`.

## Held-out measurements

Recorded in `results/evaluation-t0.8.json`. Sampling used temperature 0.8, guidance 2, and seed 42. Likelihood diagnostics cover each full split; generation metrics use the first record of each of the first 12 distinct combinations. This small deterministic sample is not a comprehensive semantic pose assessment.

| Split | NLL per pixel | Increase with mismatched caption | Generated ink IoU | Shuffled reference IoU | Blank outputs |
|---|---:|---:|---:|---:|---:|
| Unseen wording, familiar poses | 0.05912 | 0.01184 | 0.23220 | 0.14761 | 0/12 |
| Unseen pose combinations, familiar wording | 0.07728 | 0.00980 | 0.19662 | 0.15602 | 0/12 |
| Both unseen wording and pose combinations | 0.07941 | 0.00832 | 0.16492 | 0.14557 | 0/12 |

NLL is patch cross-entropy divided by four, in nats per original pixel. Lower NLL and higher IoU are better. Correct captions improve likelihood, and generated overlap beats shuffled alignment in these three small samples. This is evidence of some conditioning, not proof that each limb follows its instruction or that the model reliably composes unseen poses.

IoU compares against one specific randomized reference style; other valid drawings may score poorly. Conversely, overlap can look plausible despite incorrect poses. Teacher forcing supplies real preceding patches, so its loss is easier than free generation. These holdouts have been inspected during development; do not treat them as an untouched research benchmark. Their images still never enter gradient training, and checkpoint selection uses validation only.

## What the drawings show

`python -m text2image.text2stick.demo` runs the same fixed six ordinary-English prompts without choosing favorable samples or fixing the output. See `samples/autoregressive/plain-english-demo.png`.

The final patch model produces recognizable stick-like figures rather than the severe horizontal streaks of the single-pixel trial. However, raised arms, left/right changes, complete head outlines, and limb connectivity are not consistently correct. Cleaner targets fix a dataset defect; they do not by themselves solve pose following.

Inspect actual training images with `python -m text2image.text2stick.preview_data`. Their full captions appear in `samples/training-examples.png`. `samples/target-comparison.png` shows old and corrected targets, not neural outputs.

## Preserved earlier runs

- `archive/v1_24px/`: original low-resolution data, checkpoints, measurements, and samples.
- `archive/v2_48px_single_pixel/`: a 4,000-update trial using one token per pixel and the corrected 48-pixel targets. Its fixed demo had broken horizontal streaks despite improving teacher-forced validation. Training took about 936 seconds and peak allocated GPU memory was 1.51 GiB. That failure motivated lossless pixel-patch packing. Its checkpoint, measurements, and demo are retained for inspection.

Earlier checkpoints are not compatible with the current pixel-pattern vocabulary. No pretrained external weights or angle-model weights initialized the final model. The optional masked branch has no trained checkpoint for the current representation.

## Verification

All 24 repository regression tests passed after final changes. Coverage includes exact pixel-patch round trips and pattern orientation, normalized patch loss, autoregressive causality, cached/full-prefix equivalence, text padding, gradient flow, isolated data splits, binary outputs, and connected deterministic targets across all 108 combinations. An earlier suite run hit a CPU primitive runtime error during simultaneous training; the final full rerun passed.

The normal plain-English generation command was also run successfully using the final checkpoint. Implementation tests do not certify visual quality or semantic understanding.
