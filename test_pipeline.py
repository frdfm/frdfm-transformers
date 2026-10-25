import contextlib
import io
import runpy
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from tokenizers import Tokenizer
from model import Model
from pipeline import BOS, EOS, PAD, MAX_LEN, check_data, encode_source

class PipelineTests(unittest.TestCase):
    def test_preparation_and_tokenizer(self):
        normal = {"translation": {"en": "Hello world!", "fr": "Bonjour le monde !"}}
        long = {"translation": {"en": "hello " * 100, "fr": "bonjour"}}
        datasets = types.ModuleType("datasets")
        datasets.load_dataset = lambda *args, **kwargs: iter([normal] * 4002 + [long])
        with tempfile.TemporaryDirectory() as folder:
            script = Path(folder) / "prepare_data.py"
            shutil.copyfile(Path(__file__).with_name("prepare_data.py"), script)
            with patch.dict(sys.modules, {"datasets": datasets}), contextlib.redirect_stdout(io.StringIO()):
                runpy.run_path(str(script))
            data = Path(folder) / "data"
            check_data(data)
            tokenizer = Tokenizer.from_file(str(data / "tokenizer.json"))
            for text in ["Bonjour, l'\u00e9t\u00e9 !", "unbelievably beautiful", "\u4f60\u597d \U0001f98b", "  deux espaces"]:
                self.assertEqual(tokenizer.decode(tokenizer.encode(text).ids), text)
            self.assertEqual(encode_source(tokenizer, "Hello")[-1], EOS)
            for text in [" ", "hello " * 100]:
                with self.assertRaises(ValueError):
                    encode_source(tokenizer, text)
            for side in ["src", "tgt"]:
                rows = np.fromfile(data / f"train_{side}.bin", dtype=np.uint16).reshape(-1, MAX_LEN)
                self.assertEqual(len(rows), 2)
                self.assertTrue(np.all((rows == EOS).sum(axis=1) == 1))
            (data / "tokenizer.json").write_text("changed")
            with self.assertRaises(ValueError):
                check_data(data)

    def test_padding_invariance_and_first_token_gradient(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        model = Model().eval()
        src = torch.tensor([[4, 5, EOS]])
        padded = torch.tensor([[4, 5, EOS, PAD, PAD]])
        with torch.no_grad():
            torch.testing.assert_close(model.encoder(src), model.encoder(padded)[:, :3], atol=2e-6, rtol=2e-5)
            prefix = torch.tensor([[BOS, 6]])
            torch.testing.assert_close(model(src, prefix), model(padded, prefix), atol=3e-6, rtol=3e-5)
        target = torch.tensor([[6, 7, EOS, PAD]])
        decoder_input = torch.cat([torch.full_like(target[:, :1], BOS), target[:, :-1]], dim=1)
        self.assertEqual(decoder_input.tolist(), [[BOS, 6, 7, EOS]])
        logits = model(src, decoder_input)
        loss = torch.nn.functional.cross_entropy(logits[:, 0], target[:, 0])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(model.decoder.embed.weight.grad[BOS].abs().sum().item(), 0)

if __name__ == "__main__":
    unittest.main()
