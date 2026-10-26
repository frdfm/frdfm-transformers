import argparse
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from text2image.stick_pixels.schema import HERE,PREFIX,dataset_signature
from text2image.stick_pixels.model import PixelTransformer


def load(split):
    with np.load(HERE/"data"/f"{split}.npz") as data:
        return torch.tensor(data["captions"],dtype=torch.long),torch.tensor(data["pixels"].astype(np.int64))


def pixel_loss(model,captions,pixels):
    inputs=torch.cat([captions,pixels[:,:-1]],1)
    logits=model(inputs)[:,PREFIX-1:]
    return F.cross_entropy(logits.reshape(-1,2),pixels.reshape(-1))


@torch.inference_mode()
def evaluate(model,captions,pixels,device,batch_size=32):
    model.eval();total=0.
    for i in range(0,len(pixels),batch_size):
        total+=pixel_loss(model,captions[i:i+batch_size].to(device),pixels[i:i+batch_size].to(device)).item()*len(pixels[i:i+batch_size])
    return total/len(pixels)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--steps",type=int,default=2000)
    parser.add_argument("--batch-size",type=int,default=32)
    parser.add_argument("--seed",type=int,default=7)
    args=parser.parse_args()
    if min(args.steps,args.batch_size)<1:parser.error("Steps and batch size must be positive")
    torch.set_num_threads(2);torch.manual_seed(args.seed)
    device="cuda" if torch.cuda.is_available() else "cpu"
    if device=="cuda":torch.backends.cuda.matmul.allow_tf32=True
    dataset_hash,dataset_version=dataset_signature()
    tx,ty=load("train");vx,vy=load("val")
    model=PixelTransformer().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.0005,weight_decay=0.01)
    folder=HERE/"checkpoints";folder.mkdir(exist_ok=True)
    results=HERE/"results";results.mkdir(exist_ok=True)
    best=float("inf");history=[];started=time.time()
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters; {device}; direct binary-pixel outputs",flush=True)
    for step in range(1,args.steps+1):
        model.train()
        pick=torch.randint(len(tx),(args.batch_size,))
        lr=0.0005*min(step/100,(100/step)**0.5)
        optimizer.param_groups[0]["lr"]=lr
        loss=pixel_loss(model,tx[pick].to(device),ty[pick].to(device))
        if not torch.isfinite(loss):raise RuntimeError("Nonfinite loss")
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
        if step%200==0 or step==args.steps:
            val=evaluate(model,vx,vy,device,args.batch_size)
            row=dict(step=step,training_loss=loss.item(),validation_loss=val,seconds=time.time()-started)
            history.append(row);print(json.dumps(row),flush=True)
            if val<best:
                best=val
                state=dict(model=model.state_dict(),config=model.config,step=step,seed=args.seed,
                           validation_loss=val,representation="binary_pixels",side=24,prefix=PREFIX,
                           dataset_sha256=dataset_hash,dataset_version=dataset_version)
                temporary=folder/"model.tmp";torch.save(state,temporary);temporary.replace(folder/"model.pt")
            (results/"history.json").write_text(json.dumps(history,indent=2))
    if device=="cuda":print(f"Peak allocated GPU memory: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB",flush=True)


if __name__=="__main__":main()
