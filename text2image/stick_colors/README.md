# Colored stick figures: English to diagnostic image pixels

This experiment assigns a fixed color to each body part so that an apparently plausible silhouette cannot hide swapped or disconnected limbs. The model predicts the colored image itself. These are not recolored outputs from the black-and-white model.

| Pixel ID | Color | Meaning |
|---:|---|---|
| 0 | White | Background |
| 1 | Yellow | Head |
| 2 | Purple | Torso and neck |
| 3 | Red | Left arm |
| 4 | Blue | Right arm |
| 5 | Green | Left leg |
| 6 | Orange | Right leg |

Left/right mean **canvas-left/canvas-right**, as in the other synthetic experiments. At crossings, the top stroke owns that pixel; a pixel cannot simultaneously represent two colors. Rendering draws the body, limbs, and then head. This can divide one part's visible pixels at an overlap, so evaluation reports target diagnostics too.

## Sample outputs

The model learned these colored images from scratch. In each row below, the **left image is the actual held-out target**, and the **middle and right images are unfiltered model-generated samples** for the prompt printed above them. The color legend identifies each body part.

![Held-out colored stick-figure targets beside two generated samples per English prompt](docs/images/pose_test-comparison.png)

Snapshot: checkpoint update 4,000, sampling seed 123, temperature 0.8, guidance 2. The pictured deep-bend/raised-left-arm combinations were excluded from training. Colors make both the learned structure and remaining pose/connectivity errors visible.

[View familiar training targets beside generated samples](docs/images/train-comparison.png). Snapshot prompts, selected records, settings, and diagnostics are preserved in [held-out sample metadata](docs/images/pose_test-comparison.json) and [training sample metadata](docs/images/train-comparison.json). These documentation images are fixed snapshots, so later generation commands will not overwrite them.

## Generate and inspect

Run from the repository root in the existing venv:

```powershell
python -m text2image.stick_colors.generate "Draw a person bending forward and raising their left hand."
python -m text2image.stick_colors.compare
python -m text2image.stick_colors.compare --split pose_test
python -m text2image.stick_colors.preview_data --count 6
```

`generate` writes `samples/generated.png`. `compare` puts an actual target and two unfiltered neural samples under the exact same prompt, with a color legend. It saves `samples/train-comparison.png`, or `samples/pose_test-comparison.png` for the held-out pose split. Its JSON and NPZ files retain prompts, random seed, selected indices, and raw predictions. Training examples are familiar prompts; held-out comparisons test combinations absent from gradient training.

Ordinary English is encoded by a locally trained byte-level tokenizer. There is no phrase-to-pose parser at inference. The small synthetic language covers torso bends, arms down/out/raised, and straight/bent knees. It does not establish general English understanding. Colors are fixed by part; arbitrary instructions to change a limb's color are not taught.

## What is learned

The target is a **48 x 48 image with seven possible pixel colors**, including background. Four neighboring pixels are packed losslessly into a 2 x 2 patch ID:

```text
patch = a b        ID = a + 7*b + 49*c + 343*d
        c d

a,b,c,d are actual color IDs 0..6
patch vocabulary: 7^4 = 2401 possible patterns
begin-image input ID: 2401 (not an output category)
```

All possible palette images can be represented, even malformed figures. This fixed encoding has no knowledge of body parts, angles, joints, or valid connections.

The model shares the encoder/decoder implementation from `text2stick`, but all weights are initialized randomly for this experiment. A bidirectional two-block English encoder creates memory. A four-block causal image decoder attends to preceding pixel patches and cross-attends to English memory. Width is 96 with four attention heads. It has 1,400,257 parameters.

A batch flows through these shapes:

```text
English token IDs       [batch, up to 160]
English embeddings      [batch, text_length, 96]
English encoder memory  [batch, text_length, 96]
raw color targets       [batch, 2304]  IDs 0..6
packed target patches   [batch, 576]   IDs 0..2400
shifted decoder inputs  [batch, 576]   begin-image + patches[:-1]
decoder output logits   [batch, 576, 2401]
```

English padding is masked in encoder self-attention and decoder cross-attention. The image causal mask blocks future patches. Every target patch, including all-white zero, contributes to loss. There is no image padding to ignore.

Patch cross-entropy is divided by four to report nats per original pixel. Training weights a patch by `1 + foreground_pixels / 4`, then normalizes by the sum of patch weights. As in the binary experiment, fully specified positive captions receive a mismatched-caption ranking term, and 10% of training conditions are blanked for neural guidance. No angle, joint, or connectivity supervision is supplied to the network. The new colors provide more visible pixel supervision for body-part identity.

At inference, the model samples 576 patch IDs, unpacks their base-seven digits, and maps each pixel ID to its fixed RGB value. This is ordinary palette-image decoding. It does not infer or draw limbs, reconnect strokes, select a nearest training image, or repair an output.

## Synthetic targets and splits

The offline renderer computes skeleton coordinates from known dataset annotations, adds varied proportions/strokes/scale/position, and draws labeled strokes at 256 x 256. It downsamples a separate coverage mask for each color and assigns the dominant covered color. Interpolating numeric color IDs directly would create false colors, so the renderer does not do that.

Preparation rejects empty, disconnected, duplicate, or missing-color targets. Every accepted image contains all six body colors, with eight-connected overall foreground. This guarantee applies to targets only; generated images remain unfiltered.

There are 108 pose combinations. Training has 84; pose validation and test each reserve 12 combinations. Deep torso bend paired with raised left arm is entirely absent from training. The wording families and partial-caption policy match `text2stick`:

| Split | Image/caption records |
|---|---:|
| train | 18,816 |
| word_val | 1,344 |
| word_test | 1,344 |
| pose_val | 768 |
| pose_test | 768 |
| combined_test | 768 |

`combined_test` reuses pose-test reference images with separately held-out wording. No independent split duplicates a training raster. Train/validation/test pose combinations are assigned before rendering variations.

## Diagnostics: what counts as working

```powershell
python -m text2image.stick_colors.evaluate --temperature 0.8
```

The report includes each part separately:

- **Color IoU:** overlap between that color's generated pixels and its actual reference pixels. A red leg instead of a red arm will generally hurt this score, even if the silhouette looks human.
- **Part presence:** fraction of samples containing at least one pixel of that color. This is a weak check; it does not mean a complete limb exists.
- **Largest component fraction:** fraction of a color's pixels belonging to its largest eight-connected component. Low values expose scattered colored fragments.
- **Torso contact:** whether a non-torso color has pixels adjacent to purple, plus a more tolerant version within two pixels. A misplaced hand can also touch the torso, so this is a contact diagnostic, not proof of a correct anatomical joint or pose.
- **Connected foreground:** whether the entire nonwhite image forms one eight-connected component.

Target diagnostics are reported alongside generated diagnostics because valid occlusion and downsampling affect visible part connectivity. Color IoU is sensitive to legitimate style differences; contact alone does not establish correct pose. Shuffled reference alignment and mismatched-caption likelihood help distinguish some text conditioning from a generic colored figure prior. These small synthetic holdouts are development diagnostics, not an untouched research benchmark.

## Reproduce or resume

```powershell
python -m text2image.stick_colors.prepare_data
python -m text2image.stick_colors.preview_data
python -m text2image.stick_colors.train --steps 4000 --batch-size 16
python -m text2image.stick_colors.train --resume --steps 2000
python -m unittest -v text2image.stick_colors.test_model
```

Preparation replaces this experiment's data and tokenizer. Fresh training replaces its checkpoints. Resume requires matching data and restores the optimizer, absolute step, and saved random state. LR warms up for 200 updates to 0.0005 and then follows absolute-step inverse-square-root decay. Generation loads best-validation `checkpoints/model.pt`; resume loads `checkpoints/latest.pt`.

All earlier binary experiments remain in their own directories. See [RESULTS.md](RESULTS.md) for the measured local run and its limitations.
