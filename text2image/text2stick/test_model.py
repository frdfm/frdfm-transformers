import unittest
import torch
from text2image.text2stick.model import TextToPixels
from text2image.text2stick.train import decoder_inputs,per_image_loss


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(3)
    def test_cache_matches_full_prefix(self):
        model=TextToPixels(32,width=16,text_layers=1,pixel_layers=2,heads=2).eval()
        text=torch.tensor([[3,4,5,2,0,0],[5,6,2,0,0,0]])
        pixels=torch.randint(2,(2,9));inputs=decoder_inputs(pixels)
        with torch.no_grad():
            memory,padding=model.encode(text)
            full=model.decode(inputs,memory,padding)
            caches=[[{}, {"static":True}] for _ in model.pixel_blocks]
            pieces=[model.decode(inputs[:,i:i+1],memory,padding,i,caches) for i in range(inputs.shape[1])]
        torch.testing.assert_close(torch.cat(pieces,1),full,atol=3e-6,rtol=3e-5)
    def test_causality_padding_and_pixel_gradients(self):
        model=TextToPixels(32,width=16,text_layers=1,pixel_layers=1,heads=2).eval()
        text=torch.tensor([[3,4,2],[5,6,2]])
        padded=torch.cat([text,torch.zeros(2,3,dtype=torch.long)],1)
        pixels=torch.randint(2,(2,10));inputs=decoder_inputs(pixels)
        a=model(text,inputs);b=model(padded,inputs)
        torch.testing.assert_close(a,b,atol=3e-6,rtol=3e-5)
        changed=inputs.clone();changed[:,5:]=1-changed[:,5:]
        torch.testing.assert_close(model(text,changed)[:,:5],a[:,:5],atol=3e-6,rtol=3e-5)
        loss=per_image_loss(a,pixels).mean();loss.backward()
        self.assertGreater(model.word.weight.grad.abs().sum().item(),0)
        self.assertGreater(model.pixel.weight.grad.abs().sum().item(),0)
        self.assertEqual(a.shape,(2,10,2))

    def test_masked_pixels_and_inference_have_no_pose_parser(self):
        import builtins,importlib
        from unittest.mock import patch
        original=builtins.__import__
        def reject(name,*args,**kwargs):
            if name.startswith("text2image.stick_figures") or name.endswith("stick_pixels.prepare_data"):
                raise AssertionError("Inference attempted to import pose geometry")
            return original(name,*args,**kwargs)
        with patch("builtins.__import__",side_effect=reject):
            importlib.import_module("text2image.text2stick.generate")
            model=TextToPixels(32,width=8,text_layers=1,pixel_layers=1,heads=1,mode="masked").eval()
            text=torch.tensor([[3,4,2,0]])
            pixels=model.sample(text,temperature=0,guidance=1.)
        self.assertEqual(pixels.shape,(1,576))
        self.assertTrue(((pixels==0)|(pixels==1)).all())
        logits=torch.randn(1,6,2,requires_grad=True)
        targets=torch.tensor([[0,1,0,1,0,1]])
        mask=torch.tensor([[True,False,True,False,False,False]])
        loss=per_image_loss(logits,targets,mask=mask).mean()
        loss.backward()
        self.assertTrue(torch.all(logits.grad[~mask]==0))
        self.assertGreater(logits.grad[mask].abs().sum().item(),0)

    def test_dataset_holdouts_and_byte_tokenizer(self):
        import json,numpy as np
        from text2image.text2stick.tokenization import HERE,load_tokenizer,encode
        sets={};images={};texts={}
        for split in ("train","word_val","word_test","pose_val","pose_test","combined_test"):
            with np.load(HERE/"data"/f"{split}.npz") as data:
                self.assertEqual(set(data.files),{"text","full_text","pixels","combos","partial"})
                self.assertEqual(data["pixels"].shape[1],576)
                self.assertTrue(np.all((data["pixels"]==0)|(data["pixels"]==1)))
                sets[split]=set(map(tuple,data["combos"].tolist()))
                images[split]={np.packbits(row).tobytes() for row in data["pixels"]}
            texts[split]=set(json.loads((HERE/"data"/f"{split}-captions.json").read_text()))
        self.assertFalse(sets["train"] & sets["pose_test"])
        self.assertFalse(sets["train"] & sets["pose_val"])
        self.assertTrue(all(not(c[0]==2 and c[1]==2) for c in sets["train"]))
        for split in ("word_val","word_test","pose_val","pose_test"):
            self.assertFalse(images["train"] & images[split])
            self.assertFalse(texts["train"] & texts[split])
        self.assertEqual(sets["word_test"],sets["train"])
        tokenizer=load_tokenizer()
        # Encoding does not parse poses or reject novel words.
        self.assertTrue(encode("Draw a person waving happily.",tokenizer))
        self.assertTrue(encode("Please draw someone bending over and lifting their left hand.",tokenizer))
        self.assertTrue(encode("a person with both arms up",tokenizer))


if __name__=="__main__":unittest.main()
