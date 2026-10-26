"""Text encoder and autoregressive color-pixel patch decoder. No pose controls."""
import torch
from torch import nn
from text2image.text2stick.model import TextToPixels
from text2image.stick_colors.tokenization import PIXEL_CLASSES,PIXEL_BOS,IMAGE_TOKENS,SIDE,PATCH,COLORS
from text2image.stick_colors.pixel_tokens import unpack_tokens


class ColorPixels(TextToPixels):
    def __init__(self,vocab_size,width=96,text_layers=2,pixel_layers=4,heads=4,
                 mode="autoregressive",patch_size=PATCH,image_side=SIDE,palette_size=COLORS):
        if mode!="autoregressive" or patch_size!=PATCH or image_side!=SIDE or palette_size!=COLORS:
            raise ValueError("Incompatible color-pixel representation")
        super().__init__(vocab_size,width,text_layers,pixel_layers,heads,mode,patch_size,image_side)
        self.pixel=nn.Embedding(PIXEL_CLASSES+1,width)
        self.output=nn.Linear(width,PIXEL_CLASSES)
        self.config.update(palette_size=palette_size)

    @torch.inference_mode()
    def sample(self,text,temperature=0.8,guidance=2.,empty_text=None):
        if temperature<0:raise ValueError("Temperature must be nonnegative")
        batch=len(text)
        if guidance!=1:
            if empty_text is None:raise ValueError("Guidance requires an empty condition")
            text=torch.cat([text,empty_text.expand(batch,-1)])
        memory,padding=self.encode(text)
        caches=[[{}, {"static":True}] for _ in self.pixel_blocks]
        previous=torch.full((len(text),1),PIXEL_BOS,device=text.device,dtype=torch.long)
        result=[]
        for position in range(IMAGE_TOKENS):
            scores=self.decode(previous,memory,padding,position,caches)[:,-1].float()
            if guidance!=1:
                conditional,unconditional=scores.chunk(2)
                scores=unconditional+guidance*(conditional-unconditional)
            token=scores.argmax(-1,keepdim=True) if temperature==0 else torch.multinomial((scores/temperature).softmax(-1),1)
            result.append(token)
            previous=torch.cat([token,token]) if guidance!=1 else token
        return unpack_tokens(torch.cat(result,1))
