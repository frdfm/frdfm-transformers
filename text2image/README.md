# Text-to-image experiments

Each dataset experiment has its own package, architecture guide, training code, and generated artifacts. Run commands from the repository root with your Python environment active.

| Dataset experiment | Task | Documentation |
|---|---|---|
| [Fashion-MNIST](fashion_mnist/) | Short clothing captions to 14 x 14 or 28 x 28 grayscale images | [Full two-example forward-pass walkthrough and usage](fashion_mnist/README.md) |

## Fashion-MNIST

```powershell
python -m text2image.fashion_mnist.prepare_data --size 28
python -m text2image.fashion_mnist.train --size 28 --steps 5000
python -m text2image.fashion_mnist.generate --size 28 --all
```

Existing data and checkpoints have moved with the code. If they already exist, no new preparation or training is needed to generate images. Continue training with:

```powershell
python -m text2image.fashion_mnist.train --size 28 --resume --steps 5000
```

The existing 14 x 14 model uses the same commands with `--size 14`. All Fashion-MNIST data, checkpoints, logs, and samples are under `fashion_mnist/`; 28 x 28 artifacts retain their `28/` subfolders.

```powershell
python -m text2image.fashion_mnist.trace_batch
python -m unittest -v text2image.fashion_mnist.test_model
```

## Adding another dataset

Create a sibling package such as `text2image/<dataset_name>/`, with its own README and artifacts. Keep dataset-specific vocabulary, preprocessing, and checkpoints inside that package. Extract shared code into the parent package when more than one experiment actually needs it.
