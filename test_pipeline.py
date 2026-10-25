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

    def test_absolute_learning_rate(self):
        from train import learning_rate_at_step, LEARNING_RATE
        self.assertAlmostEqual(learning_rate_at_step(1), 5e-7)
        self.assertAlmostEqual(learning_rate_at_step(1000), LEARNING_RATE)
        self.assertAlmostEqual(learning_rate_at_step(5000), 2.2360679774997898e-4)
        self.assertAlmostEqual(learning_rate_at_step(10000), 1.5811388300841897e-4)
        self.assertLess(learning_rate_at_step(1001), learning_rate_at_step(1000))
        self.assertLess(learning_rate_at_step(5001), learning_rate_at_step(5000))
        self.assertLess(learning_rate_at_step(10001), learning_rate_at_step(10000))
        with self.assertRaises(ValueError):
            learning_rate_at_step(0)

    def test_training_loop_resume(self):
        import train
        torch.set_num_threads(2)
        rows = torch.tensor([[4, 5, EOS, PAD], [6, EOS, PAD, PAD]])
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = str(Path(folder) / "model.pt")
            logfile = str(Path(folder) / "train.log")
            with patch.multiple(train, CKPT_FILE=checkpoint, LOG_FILE=logfile,
                                DEVICE="cpu", USE_BF16=False, BATCH_SIZE=2,
                                EVAL_BATCH=2, EVAL_INTERVAL=1), \
                    patch.object(train, "check_data"), \
                    patch.object(train, "tokenizer_digest", return_value="test-tokenizer"), \
                    patch.object(train, "load_pairs", return_value=(rows, rows)), \
                    contextlib.redirect_stdout(io.StringIO()):
                with patch.object(sys, "argv", ["train.py", "--steps", "1"]):
                    train.main()
                with patch.object(sys, "argv", ["train.py", "--resume", "--steps", "1"]):
                    train.main()
            state = torch.load(checkpoint, weights_only=True, map_location="cpu")
            self.assertEqual(state["step"], 2)
            self.assertIn("optimizer", state)
            self.assertEqual(state["lr_schedule"]["name"], "inverse_sqrt")
            self.assertAlmostEqual(state["optimizer"]["param_groups"][0]["lr"], train.learning_rate_at_step(2))
            lines = Path(logfile).read_text().splitlines()
            self.assertEqual(len(lines), 3)
            self.assertEqual([line.split()[0] for line in lines[1:]], ["1", "2"])

    def test_resume_matches_uninterrupted_training(self):
        from train import save_checkpoint, restore_checkpoint
        torch.manual_seed(42)
        np.random.seed(42)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

        def update(net, opt):
            x = torch.randn(4, 3) + float(np.random.random())
            opt.zero_grad()
            net(x).square().mean().backward()
            opt.step()

        update(model, optimizer)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            save_checkpoint(path, model, optimizer, 5001, 12.0, "test-tokenizer")
            update(model, optimizer)
            expected = {k: v.clone() for k, v in model.state_dict().items()}
            resumed = torch.nn.Linear(3, 2)
            resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.1)
            result = restore_checkpoint(path, resumed, resumed_optimizer, "test-tokenizer")
            self.assertEqual(result, (5001, 12.0))
            update(resumed, resumed_optimizer)
            for key, value in resumed.state_dict().items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
            with self.assertRaises(ValueError):
                restore_checkpoint(path, resumed, resumed_optimizer, "wrong-tokenizer")
            torch.save({"model": expected, "format_version": 2,
                        "tokenizer_sha256": "test-tokenizer"}, path)
            with self.assertRaises(ValueError):
                restore_checkpoint(path, resumed, resumed_optimizer, "test-tokenizer")
            self.assertEqual(restore_checkpoint(path, resumed, resumed_optimizer,
                                              "test-tokenizer", 5000), (5000, 0.0))

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
