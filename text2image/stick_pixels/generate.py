"""Inference imports no skeleton, angle, joint, or source-image renderer."""
import argparse
from pathlib import Path
import torch
from PIL import Image,ImageDraw
from text2image.stick_pixels.schema import HERE,SIDE,encode
from text2image.stick_pixels.model import PixelTransformer


def save_grid(pixels,path,labels=None,columns=4):
    array=pixels.detach().cpu().numpy().reshape(-1,SIDE,SIDE)
    cell=192
    row_height=cell+58
    panels=len(array)
    rows=(panels+columns-1)//columns
    image=Image.new("RGB",(columns*cell,rows*row_height),"#eeeeee")
    pen=ImageDraw.Draw(image)
    for i,panel in enumerate(array):
        x,y=(i%columns)*cell,(i//columns)*row_height
        raster=Image.fromarray(((1-panel)*255).astype("uint8"))
        image.paste(raster.resize((cell,cell),Image.Resampling.NEAREST),(x,y+50))
        if labels:
            clauses=labels[i].split(" ; ")
            for j in range(0,len(clauses),2):
                pen.text((x+3,y+2+12*(j//2)),"; ".join(clauses[j:j+2]),fill="black")
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);image.save(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--temperature",type=float,default=0.8)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--count",type=int,default=4)
    parser.add_argument("--output",type=Path,default=HERE/"samples"/"generated.png")
    args=parser.parse_args()
    if args.count<1 or args.temperature<0:parser.error("Count must be positive; temperature nonnegative")
    try:caption=encode(args.prompt)
    except ValueError as error:parser.error(str(error))
    device="cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    state=torch.load(HERE/"checkpoints"/"model.pt",weights_only=True,map_location=device)
    model=PixelTransformer(**state["config"]).to(device).eval()
    model.load_state_dict(state["model"])
    images=model.sample(torch.tensor([caption]*args.count,device=device),args.temperature)
    save_grid(images,args.output,[args.prompt]*args.count,columns=min(4,args.count))
    print(f"Checkpoint training dataset version: {state.get('dataset_version',1)}")
    print(args.output.resolve())


if __name__=="__main__":main()
