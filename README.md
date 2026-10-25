# Small Transformers trained from scratch

Two independent educational experiments:

| Folder | Experiment | Documentation |
|---|---|---|
| `en2fr/` | English-to-French translation | [Architecture and usage](en2fr/README.md) |
| `text2image/` | Text-conditioned grayscale clothing generation | [Architecture and usage](text2image/README.md) |

Run from the repository root with your Python environment active:

```powershell
python -m en2fr.translate "the weather is nice today"
python -m en2fr.train --resume --steps 5000
python -m text2image.generate --size 28 "draw a bag"
```

The translator's existing data, checkpoints, and logs have moved into `en2fr/`. No retraining or data preparation is needed for this move. Each experiment keeps its own generated artifacts.

Run both test suites:

```powershell
python -m unittest -v en2fr.test_pipeline text2image.test_model
```
