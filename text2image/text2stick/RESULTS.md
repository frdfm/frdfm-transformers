# Local Text2Stick trial

This is an initial working plain-English-to-pixel experiment, not a demonstrated solution to compositional image generation. The inference path uses a learned text encoder and pixel decoder. Offline target generation uses the synthetic renderer; inference does not.

## Training run

- 18,816 training image/caption pairs, covering 84 pose combinations with varied styles.
- 937,250 parameters, 24 x 24 binary pixels, batch size 32.
- Trained from scratch for 4,000 optimizer updates with seed 7 on the local CUDA GPU.
- About 219 seconds for training; peak allocated GPU memory about 0.79 GiB. This excludes dataset preparation and evaluation, and is allocated memory rather than total GPU reservation.
- Best validation checkpoint: update 3,500; average validation pixel NLL 0.10175 on the fixed selection subsets. Latest resumable checkpoint: update 4,000.
- Validation selected the checkpoint before final test evaluation. No test result selected model weights.

## Held-out measurements

Recorded in `results/evaluation-t0.8.json`. Generation used temperature 0.8, guidance 2, and 12 fixed examples per split. Pixel loss uses the full split with teacher forcing; generated-image metrics use those 12 examples only.

| Split | Pixel NLL | NLL increase with mismatched caption | Generated ink IoU | Shuffled reference IoU | Blank outputs |
|---|---:|---:|---:|---:|---:|
| Unseen wording, familiar poses | 0.10386 | 0.01128 | 0.28309 | 0.25422 | 0/12 |
| Unseen pose combinations, familiar wording | 0.14979 | 0.01280 | 0.27912 | 0.25768 | 0/12 |
| Both unseen wording and pose combinations | 0.15318 | 0.01101 | 0.24628 | 0.28721 | 0/12 |

Lower NLL is better; higher IoU is better. Correct captions improve teacher-forced image likelihood slightly. Generated pixel overlap beats shuffled alignment in the first two small samples, but loses in the combined holdout. **These results do not establish reliable compositional generalization.** A much larger generated sample and semantic pose assessment would be needed for that claim.

IoU compares ink against one particular randomized reference drawing; an alternative legitimate style can score poorly. Conversely, plausible overlap does not prove every requested limb is correct. Teacher forcing supplies real preceding pixels, so its loss can look good even when fully generated figures are malformed.

## Visual inspection

Run `python -m text2image.text2stick.demo` to reproduce the fixed six-prompt grid at `samples/autoregressive/plain-english-demo.png`. Prompts are recorded in `samples/demo-prompts.json`. These outputs are not repaired or selected from a larger pool.

Some drawings resemble figures, but disconnected limbs, thick strokes, and incorrect poses remain common. In the fixed demo, changing left/right wording does not reliably produce the requested mirrored arm behavior. This is a useful version to test, not a claim of unrestricted English understanding.

## Masked trial

A separate optional masked-pixel model was initialized from the local autoregressive checkpoint and trained for 2,500 updates. Validation selected update 750. Greedy generated images were often nearly empty; mean ink IoU across the three test samples was approximately 0.034, 0.124, and 0.132. This branch remains optional and is not the default. Its fully masked pixel loss measures a different task from autoregressive teacher-forced NLL.

## Verification

All 22 repository regression tests passed, including four Text2Stick tests covering cache equivalence, causality, text padding, gradients, isolated data splits, byte tokenization, and masked-pixel supervision. Tests check implementation behavior; they do not certify visual quality or pose understanding.
