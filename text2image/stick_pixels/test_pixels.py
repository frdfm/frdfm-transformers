import unittest
from unittest.mock import patch
import numpy as np
import torch
from text2image.stick_pixels.schema import encode,PAD,PREFIX,PIXELS
from text2image.stick_pixels.model import PixelTransformer
from text2image.stick_pixels.train import pixel_loss


class PixelTests(unittest.TestCase):
    def test_pixels_masks_alignment_and_gradients(self):
        torch.set_num_threads(2)
        model=PixelTransformer(width=16,layers=1,heads=2).eval()
        caption="torso deeply bent ; left arm raised ; right arm down ; left knee bent ; right knee straight"
        text=torch.tensor([encode(caption)]*2)
        pixels=torch.randint(2,(2,PIXELS))
        tokens=torch.cat([text,pixels[:,:-1]],1)
        logits=model(tokens);logits.retain_grad()
        self.assertEqual(logits[:,PREFIX-1:].shape,(2,PIXELS,2))
        changed=tokens.clone();changed[:,PREFIX+3:]=1-changed[:,PREFIX+3:]
        with torch.no_grad():
            torch.testing.assert_close(model(tokens)[:,:PREFIX+3],model(changed)[:,:PREFIX+3])
        loss=torch.nn.functional.cross_entropy(logits[:,PREFIX-1:].reshape(-1,2),pixels.reshape(-1))
        torch.testing.assert_close(loss,pixel_loss(model,text,pixels))
        loss.backward()
        self.assertTrue(torch.all(logits.grad[:,:PREFIX-1]==0))
        self.assertGreater(logits.grad[:,PREFIX-1].abs().sum().item(),0)
        self.assertEqual(model.output.out_features,2)
        self.assertFalse(torch.any(text==0) or torch.any(text==1))

    def test_inference_imports_no_pose_renderer(self):
        import builtins
        import importlib
        original_import = builtins.__import__
        def reject_pose_import(name, *args, **kwargs):
            if name.startswith("text2image.stick_figures"):
                raise AssertionError("Pixel inference imported angle/skeleton implementation")
            return original_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=reject_pose_import):
            module = importlib.import_module("text2image.stick_pixels.generate")
            self.assertIs(module.PixelTransformer, PixelTransformer)
            model = PixelTransformer(width=8, layers=1, heads=1).eval()
            logits = model(torch.tensor([encode("left arm raised")]))
            self.assertEqual(logits.shape[-1], 2)

    def test_randomized_generator_is_seeded_and_diverse(self):
        from text2image.stick_pixels.prepare_data import raster
        for combo in [(0,0,0,0,0),(2,2,2,1,1),(1,1,2,0,1)]:
            a=raster(combo,np.random.default_rng(12))
            b=raster(combo,np.random.default_rng(12))
            np.testing.assert_array_equal(a,b)
            self.assertEqual(a.shape,(PIXELS,))
            self.assertTrue(np.all((a==0)|(a==1)))
            self.assertGreater(int(a.sum()),10)
        variants={raster((2,2,0,0,0),np.random.default_rng(i)).tobytes() for i in range(32)}
        self.assertGreaterEqual(len(variants),30)

    def test_new_training_checkpoint_records_data_identity(self):
        import contextlib, io, sys, tempfile
        from pathlib import Path
        from text2image.stick_pixels import train
        from text2image.stick_pixels.schema import dataset_signature
        signature, version = dataset_signature()
        self.assertEqual(version,2)
        self.assertEqual(len(signature),64)
        text=torch.tensor([encode("left arm raised")]*2)
        pixels=torch.zeros((2,PIXELS),dtype=torch.long)
        pixels[:,100:110]=1
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(train,"HERE",Path(folder)), \
                    patch.object(train,"load",return_value=(text,pixels)), \
                    patch.object(train,"dataset_signature",return_value=(signature,version)), \
                    patch.object(train,"PixelTransformer",side_effect=lambda: PixelTransformer(width=8,layers=1,heads=1)), \
                    patch.object(torch.cuda,"is_available",return_value=False), \
                    patch.object(sys,"argv",["train","--steps","1","--batch-size","2"]), \
                    contextlib.redirect_stdout(io.StringIO()):
                train.main()
            state=torch.load(Path(folder)/"checkpoints"/"model.pt",map_location="cpu",weights_only=True)
            self.assertEqual(state["dataset_sha256"],signature)
            self.assertEqual(state["dataset_version"],2)

    def test_data_contains_no_pose_targets_or_split_leakage(self):
        from text2image.stick_pixels.schema import HERE
        manifest=__import__("json").loads((HERE/"data"/"manifest.json").read_text())
        groups={n:set(map(tuple,rows)) for n,rows in manifest["combinations"].items()}
        for a,b in [("train","val"),("train","test"),("val","test")]:
            self.assertFalse(groups[a]&groups[b])
        image_sets={}
        for name in ("train","val","test"):
            with np.load(HERE/"data"/f"{name}.npz") as d:
                image_sets[name]={np.packbits(row).tobytes() for row in d["pixels"]}
        for a,b in [("train","val"),("train","test"),("val","test")]:
            self.assertFalse(image_sets[a]&image_sets[b])
        self.assertTrue(all(not(c[0]==2 and c[1]==2) for c in groups["train"]))
        for split in ("train","val","test"):
            with np.load(HERE/"data"/f"{split}.npz") as d:
                self.assertEqual(set(d.files),{"pixels","captions","combos"})
                self.assertEqual(d["pixels"].shape[1],PIXELS)
                self.assertTrue(np.all((d["pixels"]==0)|(d["pixels"]==1)))
                self.assertTrue(np.all(d["pixels"].sum(1)>0))
                if manifest.get("format_version")==2:
                    variants=manifest["variants_per_split"][split]
                    self.assertEqual(len(d["pixels"]),len(groups[split])*variants)
                    self.assertEqual(len(image_sets[split]),len(d["pixels"]))


if __name__=="__main__":unittest.main()
