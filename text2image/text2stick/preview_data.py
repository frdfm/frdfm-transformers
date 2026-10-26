"""Show actual training targets and their complete captions; no neural inference."""
import argparse,json
import numpy as np
import torch
from text2image.text2stick.tokenization import HERE
from text2image.text2stick.generate import save_grid


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--count",type=int,default=12)
    parser.add_argument("--seed",type=int,default=42)
    args=parser.parse_args()
    with np.load(HERE/"data"/"train.npz") as data:
        if not 1<=args.count<=len(data["pixels"]):parser.error("Count must fit the training dataset")
        indices=np.random.default_rng(args.seed).choice(len(data["pixels"]),args.count,replace=False)
        pixels=torch.tensor(data["pixels"][indices].astype(np.int64))
    captions=json.loads((HERE/"data"/"train-captions.json").read_text())
    path=HERE/"samples"/"training-examples.png"
    save_grid(pixels,path,[f"Train #{i}: {captions[i]}" for i in indices],columns=min(3,args.count))
    print(path.resolve())


if __name__=="__main__":main()
