# Tiny clothing image Transformer

Train a small caption-conditioned image generator entirely from scratch. No pretrained text encoder, VAE, diffusion model, or external inference API.

**Scope:** 28 x 28 grayscale clothing silhouettes (14 x 14 also supported), with 16 intensity levels and ten Fashion-MNIST categories. These are low-resolution product silhouettes, not free-form sketches or general text-to-image. Displaying them larger does not add detail. The text vocabulary is learned from short synthetic captions; natural-language understanding is limited to those captions and category words.

The default 28 x 28 network has 901,392 parameters: four causal Transformer blocks, width 128, and four attention heads. It reads a ten-position padded caption and predicts 784 pixels in row order. Training is exactly next-token prediction, as in a language model. Each output is sampled rather than retrieved from the dataset. At training time the last caption position predicts the first pixel; the remaining pixels use teacher forcing.

## Run from the repository root, in your existing venv

```powershell
python -m text2image.prepare_data --size 28
python -m text2image.train --size 28 --steps 5000
python -m text2image.generate "draw a bag"
python -m text2image.generate --all
```

Preparation downloads about 30 MB of compressed Fashion-MNIST from the dataset authors' repository and verifies the published MD5 checksums. It preserves the native 28 x 28 resolution and quantizes intensities to 16 values. With `--size 14` it instead averages 2 x 2 pixel blocks. A seeded 5,000-image validation split is held out from the original 60,000 training examples; the official 10,000-image test set is prepared but never used to choose checkpoints.

Verified on your RTX 3060 12 GB: a five-update 28 x 28 CUDA smoke test at batch size 64 peaked at **4.53 GiB** of PyTorch-allocated memory (total device use can be higher). The 14 x 14 version used 0.68 GiB. At 28 x 28 there are four times as many pixels, and attention score work grows roughly sixteen-fold. Sampling is also slower because pixels are generated sequentially without a KV cache. Batch size 64 is the tested default; use 32 if other workloads need the GPU.

The prepared 28 x 28 data and its five-step smoke checkpoint are in separate subfolders. Start fresh for useful images; do not resume the smoke checkpoint. Existing 14 x 14 training is preserved. Training time and quality depend on the run; inspect samples and validation loss rather than assuming a fixed step count guarantees quality. Higher resolution preserves details but does not by itself fix category mistakes.


## Prompts and output

Categories: `t-shirt`, `trousers`, `pullover`, `dress`, `coat`, `sandal`, `shirt`, `sneaker`, `bag`, `ankle boot`.

Training templates: `draw a {category}`, `a drawing of a {category}`, `a grayscale image of a {category}`, or the category alone. Prefer these forms; the model has not learned attributes such as red, striped, or two objects. Unknown words are rejected. Some categories, especially shirt/top/pullover/coat, overlap visually.

```powershell
python -m text2image.generate "a drawing of a sneaker" --count 8 --temperature 0.8 --seed 12
```

Generation writes `text2image/samples/28/generated.png`. Images are inverted for dark shapes on a white background, and enlarged using nearest-neighbor display. `--all` makes four samples per category. Change the seed for different samples; lowering temperature usually reduces variation.

Training writes a labeled ten-category preview at every evaluation, CSV losses, latest checkpoint `checkpoints/28/model.pt`, and lowest-validation-loss checkpoint `checkpoints/28/best.pt`. Data, samples, logs, and checkpoints stay under this folder. Generation defaults to the latest checkpoint.

## Resume

```powershell
python -m text2image.train --size 28 --resume --steps 5000
```

This adds updates, restores AdamW and random states, and appends logs. The LR warms up to 0.0003 over 500 absolute steps, then decays with the inverse square root of the absolute step. It never restarts on resume. A fresh run without `--resume` replaces the training checkpoint and log.

For a quick plumbing check, use `--steps 2 --batch-size 2 --eval-every 1 --val-batches 1 --no-samples`. This trains the same checkpoint path, so do it before a real run. By default validation uses all 5,000 held-out examples; `--val-batches` caps evaluation for quick experiments.

## Tests and references

```powershell
python -m unittest -v text2image.test_model
```

Dataset and checksums: https://github.com/zalandoresearch/fashion-mnist (MIT license). Pixel-sequence modeling is inspired by Image GPT: https://github.com/openai/image-gpt. This is an independent small educational implementation, not a reproduction of its architecture or results.

## Existing 14 x 14 model

All entry points accept `--size 14` or `--size 28` (default 28). Resolution is saved in new model configs; older checkpoints without it remain 14 x 14. The old data/checkpoint/log/sample paths are preserved for size 14. Size 28 uses a `28/` subfolder under each. A model trained at 14 x 14 cannot be resumed directly as 28 x 28 because its position embeddings have a different shape; train the new resolution from scratch.

```powershell
python -m text2image.generate --size 14 --all
python -m text2image.train --size 14 --resume --steps 5000
```
