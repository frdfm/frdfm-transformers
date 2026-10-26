"""Text encoder + cross-attending pixel decoder; no geometry/pose prediction."""
import torch
from torch import nn
import torch.nn.functional as F
from text2image.text2stick.pixel_tokens import unpack_tokens
from text2image.text2stick.tokenization import MAX_TEXT,IMAGE_TOKENS,PIXEL_CLASSES,PIXEL_BOS,PAD,SIDE,PATCH


class Attention(nn.Module):
    def __init__(self,width,heads):
        super().__init__()
        self.heads,self.depth=heads,width//heads
        self.q,self.k,self.v,self.out=[nn.Linear(width,width) for _ in range(4)]

    def project(self,x,layer):
        b,t,_=x.shape
        return layer(x).reshape(b,t,self.heads,self.depth).transpose(1,2)

    def forward(self,x,context=None,padding=None,causal=False,cache=None):
        context=x if context is None else context
        q=self.project(x,self.q)
        if cache is not None and cache.get("static") and "k" in cache:
            k,v=cache["k"],cache["v"]
        else:
            k,v=self.project(context,self.k),self.project(context,self.v)
            if cache is not None:
                if "k" in cache and not cache.get("static"):
                    k=torch.cat([cache["k"],k],dim=2);v=torch.cat([cache["v"],v],dim=2)
                cache.update(k=k,v=v)
        mask=None
        if padding is not None:
            mask=torch.zeros(padding.shape[0],1,1,padding.shape[1],device=x.device,dtype=q.dtype)
            mask=mask.masked_fill(padding[:,None,None,:],float("-inf"))
        h=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,is_causal=causal and cache is None)
        h=h.transpose(1,2).reshape(x.shape[0],x.shape[1],-1)
        return self.out(h)


class TextBlock(nn.Module):
    def __init__(self,width,heads):
        super().__init__()
        self.n1,self.n2=nn.LayerNorm(width),nn.LayerNorm(width)
        self.attn=Attention(width,heads)
        self.ff=nn.Sequential(nn.Linear(width,width*4),nn.GELU(),nn.Linear(width*4,width))
    def forward(self,x,padding):
        h=self.n1(x);x=x+self.attn(h,padding=padding)
        return x+self.ff(self.n2(x))


class PixelBlock(nn.Module):
    def __init__(self,width,heads):
        super().__init__()
        self.n1,self.n2,self.n3=nn.LayerNorm(width),nn.LayerNorm(width),nn.LayerNorm(width)
        self.self_attn,self.cross_attn=Attention(width,heads),Attention(width,heads)
        self.ff=nn.Sequential(nn.Linear(width,width*4),nn.GELU(),nn.Linear(width*4,width))
    def forward(self,x,memory,padding,cache=None,causal=True):
        x=x+self.self_attn(self.n1(x),causal=causal,cache=cache[0] if cache is not None else None)
        x=x+self.cross_attn(self.n2(x),memory,padding,cache=cache[1] if cache is not None else None)
        return x+self.ff(self.n3(x))


class TextToPixels(nn.Module):
    def __init__(self,vocab_size,width=96,text_layers=2,pixel_layers=4,heads=4,mode="autoregressive",patch_size=PATCH,image_side=SIDE):
        super().__init__()
        if patch_size!=PATCH or image_side!=SIDE:raise ValueError("Checkpoint pixel representation differs from this code")
        self.mode=mode
        self.config=dict(vocab_size=vocab_size,width=width,text_layers=text_layers,pixel_layers=pixel_layers,heads=heads,mode=mode,patch_size=patch_size,image_side=image_side)
        self.word=nn.Embedding(vocab_size,width,padding_idx=PAD)
        self.text_position=nn.Embedding(MAX_TEXT,width)
        self.text_blocks=nn.ModuleList([TextBlock(width,heads) for _ in range(text_layers)])
        self.text_norm=nn.LayerNorm(width)
        self.pixel=nn.Embedding(PIXEL_CLASSES+1,width)
        self.pixel_position=nn.Embedding(IMAGE_TOKENS,width)
        self.pixel_blocks=nn.ModuleList([PixelBlock(width,heads) for _ in range(pixel_layers)])
        self.pixel_norm=nn.LayerNorm(width)
        self.output=nn.Linear(width,PIXEL_CLASSES)

    def encode(self,text):
        length=max(1,int((text!=PAD).sum(1).max()))
        text=text[:,:length]
        padding=text==PAD
        x=self.word(text)+self.text_position.weight[:text.shape[1]]
        for block in self.text_blocks:x=block(x,padding)
        return self.text_norm(x),padding

    def decode(self,inputs,memory,padding,position=0,caches=None):
        x=self.pixel(inputs)+self.pixel_position.weight[position:position+inputs.shape[1]]
        for i,block in enumerate(self.pixel_blocks):
            x=block(x,memory,padding,caches[i] if caches is not None else None,causal=self.mode=="autoregressive")
        return self.output(self.pixel_norm(x))

    def forward(self,text,inputs):
        memory,padding=self.encode(text)
        return self.decode(inputs,memory,padding)

    @torch.inference_mode()
    def sample(self,text,temperature=0.8,guidance=1.0,empty_text=None):
        if temperature<0:raise ValueError("Temperature must be nonnegative")
        if self.mode=="masked":
            return self.sample_masked(text,temperature,guidance,empty_text)
        batch=len(text)
        if guidance!=1:
            if empty_text is None:raise ValueError("Guidance needs an empty-text condition")
            text=torch.cat([text,empty_text.expand(batch,-1)],0)
        memory,padding=self.encode(text)
        caches=[[{}, {"static":True}] for _ in self.pixel_blocks]
        previous=torch.full((len(text),1),PIXEL_BOS,device=text.device,dtype=torch.long)
        result=[]
        for position in range(IMAGE_TOKENS):
            scores=self.decode(previous,memory,padding,position,caches)[:,-1].float()
            if guidance!=1:
                conditional,unconditional=scores.chunk(2)
                scores=unconditional+guidance*(conditional-unconditional)
            next_pixel=scores.argmax(-1,keepdim=True) if temperature==0 else torch.multinomial((scores/temperature).softmax(-1),1)
            result.append(next_pixel)
            previous=torch.cat([next_pixel,next_pixel],0) if guidance!=1 else next_pixel
        return unpack_tokens(torch.cat(result,1))


    @torch.inference_mode()
    def sample_masked(self,text,temperature=0.8,guidance=2.,empty_text=None,steps=16):
        import math
        batch=len(text)
        if guidance!=1:
            if empty_text is None:raise ValueError("Guidance needs an empty text condition")
            text=torch.cat([text,empty_text.expand(batch,-1)],0)
        memory,padding=self.encode(text)
        caches=[[None,{"static":True}] for _ in self.pixel_blocks]
        pixels=torch.full((batch,IMAGE_TOKENS),PIXEL_BOS,device=text.device,dtype=torch.long)
        for step in range(steps):
            inputs=torch.cat([pixels,pixels],0) if guidance!=1 else pixels
            scores=self.decode(inputs,memory,padding,caches=caches).float()
            if guidance!=1:
                conditional,unconditional=scores.chunk(2)
                scores=unconditional+guidance*(conditional-unconditional)
            probabilities=(scores/(temperature if temperature>0 else 1.)).softmax(-1)
            chosen=scores.argmax(-1) if temperature==0 else torch.multinomial(probabilities.reshape(-1,PIXEL_CLASSES),1).reshape(batch,IMAGE_TOKENS)
            confidence=probabilities.gather(-1,chosen.unsqueeze(-1)).squeeze(-1)
            confidence=confidence.masked_fill(pixels!=PIXEL_BOS,-torch.inf)
            remaining=int(IMAGE_TOKENS*math.cos((step+1)/steps*math.pi/2)) if step+1<steps else 0
            for b in range(batch):
                number=max(0,int((pixels[b]==PIXEL_BOS).sum())-remaining)
                if number:
                    indices=confidence[b].topk(number).indices
                    pixels[b,indices]=chosen[b,indices]
        return unpack_tokens(pixels)
