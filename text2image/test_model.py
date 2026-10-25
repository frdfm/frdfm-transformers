import tempfile
from pathlib import Path
import unittest
import torch
from PIL import Image
from text2image.model import ImageTransformer, encode_prompt, CLASSES, TEMPLATES, PREFIX, PIXELS, LEVELS
from text2image.train import pixel_loss
from text2image.generate import save_grid


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(1)

    def test_prompts(self):
        for template in TEMPLATES:
            for category in CLASSES:
                self.assertEqual(len(encode_prompt(template.format(category))), PREFIX)
        for invalid in ["", "a purple spaceship"]:
            with self.assertRaises(ValueError):
                encode_prompt(invalid)

    def test_causality_and_pixel_alignment(self):
        model = ImageTransformer(width=32, layers=1, heads=4).eval()
        caption = torch.tensor([encode_prompt("draw a bag")])
        pixels = torch.randint(LEVELS, (1, PIXELS))
        tokens = torch.cat([caption, pixels[:, :-1]], dim=1)
        altered = tokens.clone()
        altered[:, PREFIX + 5:] = 0
        with torch.no_grad():
            a, b = model(tokens), model(altered)
        torch.testing.assert_close(a[:, :PREFIX + 5], b[:, :PREFIX + 5])
        self.assertEqual(a[:, PREFIX - 1:].shape, (1, PIXELS, LEVELS))
        loss = pixel_loss(model, caption, pixels)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(model.embed.weight.grad.abs().sum().item(), 0)

    def test_native_resolution(self):
        model = ImageTransformer(width=16, layers=1, heads=2, side=28).eval()
        captions = torch.tensor([encode_prompt("draw a bag")])
        pixels = torch.randint(LEVELS, (1, 784))
        loss = pixel_loss(model, captions, pixels)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(model.position.num_embeddings, PREFIX + 784)
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "native.png"
            save_grid(pixels.reshape(1, 28, 28), file, columns=1, scale=6)
            with Image.open(file) as image:
                self.assertEqual(image.size, (168, 188))

    def test_sampling_and_png(self):
        model = ImageTransformer(width=16, layers=1, heads=2).eval()
        result = model.sample(torch.tensor([encode_prompt("draw a bag")]))
        self.assertEqual(result.shape, (1, 14, 14))
        self.assertTrue(((result >= 0) & (result < LEVELS)).all())
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "sample.png"
            save_grid(result, file, columns=1, labels=["bag"])
            with Image.open(file) as image:
                self.assertEqual(image.size, (168, 188))


if __name__ == "__main__":
    unittest.main()
