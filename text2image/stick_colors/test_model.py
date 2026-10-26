import unittest
import numpy as np
import torch
from text2image.stick_colors.tokenization import PIXELS,PIXEL_CLASSES,IMAGE_TOKENS,PIXEL_BOS,HERE
from text2image.stick_colors.pixel_tokens import pack_pixels,unpack_tokens
from text2image.stick_colors.model import ColorPixels
from text2image.stick_colors.train import per_image_loss,decoder_inputs
from text2image.stick_colors.metrics import diagnose


class ColorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(3)

    def test_all_palette_patterns_round_trip_and_loss(self):
        tokens=(torch.arange(IMAGE_TOKENS*5)%PIXEL_CLASSES).reshape(5,IMAGE_TOKENS)
        pixels=unpack_tokens(tokens)
        self.assertEqual(pixels.shape,(5,PIXELS))
        self.assertTrue(((pixels>=0)&(pixels<7)).all())
        torch.testing.assert_close(pack_pixels(pixels),tokens)
        grid=unpack_tokens(torch.full((1,IMAGE_TOKENS),1+2*7+3*49+4*343)).reshape(48,48)
        self.assertEqual(grid[:2,:2].tolist(),[[1,2],[3,4]])
        scores=torch.randn(2,6,PIXEL_CLASSES,requires_grad=True)
        target=torch.randint(PIXEL_CLASSES,(2,6))
        expected=torch.nn.functional.cross_entropy(scores.reshape(-1,PIXEL_CLASSES),target.reshape(-1))/4
        actual=per_image_loss(scores,target).mean()
        torch.testing.assert_close(actual,expected)
        actual.backward();self.assertGreater(scores.grad.abs().sum().item(),0)

    def test_attention_cache_causality_padding_and_color_gradients(self):
        model=ColorPixels(32,width=16,text_layers=1,pixel_layers=1,heads=2).eval()
        text=torch.tensor([[3,4,1,0],[5,1,0,0]])
        targets=torch.randint(PIXEL_CLASSES,(2,8));inputs=decoder_inputs(targets)
        self.assertTrue((inputs[:,0]==PIXEL_BOS).all())
        full=model(text,inputs)
        memory,padding=model.encode(text)
        caches=[[{}, {"static":True}] for _ in model.pixel_blocks]
        cached=torch.cat([model.decode(inputs[:,i:i+1],memory,padding,i,caches) for i in range(8)],1)
        torch.testing.assert_close(full,cached,atol=3e-6,rtol=3e-5)
        changed=inputs.clone();changed[:,4:]=(changed[:,4:]+1)%PIXEL_CLASSES
        torch.testing.assert_close(full[:,:4],model(text,changed)[:,:4],atol=3e-6,rtol=3e-5)
        padded=torch.cat([text,torch.zeros(2,3,dtype=torch.long)],1)
        torch.testing.assert_close(full,model(padded,inputs),atol=3e-6,rtol=3e-5)
        per_image_loss(full,targets).mean().backward()
        self.assertGreater(model.word.weight.grad.abs().sum().item(),0)
        self.assertGreater(model.output.weight.grad.abs().sum().item(),0)

    def test_generation_is_palette_pixels_without_renderer(self):
        import builtins
        from unittest.mock import patch
        original=builtins.__import__
        def reject(name,*args,**kwargs):
            if name.startswith('text2image.stick_figures') or name.endswith('.renderer'):
                raise AssertionError('Generation attempted to import pose geometry')
            return original(name,*args,**kwargs)
        model=ColorPixels(16,width=8,text_layers=1,pixel_layers=1,heads=1).eval()
        with patch('builtins.__import__',side_effect=reject):
            result=model.sample(torch.tensor([[3,1,0]]),temperature=0,guidance=1)
        self.assertEqual(result.shape,(1,PIXELS))
        self.assertTrue(((result>=0)&(result<7)).all())

    def test_metrics_detect_disconnected_colored_limb(self):
        image=np.zeros((48,48),dtype=np.int64)
        image[10:35,24]=2;image[9,24]=1
        image[15,14:24]=3;image[15,25:35]=4
        image[35:40,23]=5;image[35:40,25]=6
        good=diagnose(image[None],image[None])
        self.assertEqual(good['mean_foreground_color_iou'],1)
        self.assertEqual(good['connected_foreground_fraction'],1)
        self.assertEqual(good['parts']['left arm']['torso_contact_within_2px_fraction'],1)
        broken=image.copy();broken[broken==3]=0;broken[2,2:12]=3
        bad=diagnose(broken[None],image[None])
        self.assertEqual(bad['connected_foreground_fraction'],0)
        self.assertEqual(bad['parts']['left arm']['torso_contact_within_2px_fraction'],0)
        self.assertEqual(bad['parts']['left arm']['mean_color_iou'],0)

    def test_dataset_colors_connectivity_and_no_leakage(self):
        from text2image.stick_colors.renderer import is_connected
        images={};combos={}
        for split in ('train','word_val','word_test','pose_val','pose_test','combined_test'):
            with np.load(HERE/'data'/f'{split}.npz') as data:
                pixels=data['pixels']
                self.assertEqual(pixels.shape[1],PIXELS)
                self.assertTrue(((pixels>=0)&(pixels<7)).all())
                for raster in pixels[::max(1,len(pixels)//12)]:
                    self.assertEqual(set(np.unique(raster)),set(range(7)))
                    self.assertTrue(is_connected(raster))
                images[split]={row.tobytes() for row in pixels}
                self.assertEqual(len(images[split]),len(pixels))
                combos[split]=set(map(tuple,data['combos'].tolist()))
        for split in ('word_val','word_test','pose_val','pose_test','combined_test'):
            self.assertFalse(images['train']&images[split])
        self.assertFalse(combos['train']&combos['pose_test'])
        self.assertFalse(combos['train']&combos['pose_val'])
        self.assertTrue(all(not(c[0]==2 and c[1]==2) for c in combos['train']))


if __name__=='__main__':unittest.main()
