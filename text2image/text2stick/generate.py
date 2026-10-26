"""Plain English -> learned pixel probabilities -> PNG. No semantic rule parser."""
import argparse
import hashlib
from pathlib import Path
import torch
from PIL import Image,ImageDraw
from text2image.text2stick.tokenization import HERE,SIDE,MAX_TEXT,encode,load_tokenizer
from text2image.text2stick.model import TextToPixels


def save_grid(pixels,path,labels=None,columns=4):
    arrays=pixels.detach().cpu().numpy().reshape(-1,SIDE,SIDE)
    import textwrap
    size=192;rows=(len(arrays)+columns-1)//columns
    lines=[textwrap.wrap(label,width=30) for label in labels] if labels else [[] for _ in arrays]
    heading=max(20,max((len(row) for row in lines),default=0)*12+8)
    image=Image.new("RGB",(columns*size,rows*(size+heading+4)),"#eeeeee")
    draw=ImageDraw.Draw(image)
    for i,array in enumerate(arrays):
        x,y=i%columns*size,i//columns*(size+heading+4)
        raster=Image.fromarray(((1-array)*255).astype("uint8"))
        image.paste(raster.resize((size,size),Image.Resampling.NEAREST),(x,y+heading))
        for j,line in enumerate(lines[i]):draw.text((x+3,y+2+j*12),line,fill="black")
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);image.save(path)


def load_model(device,mode="autoregressive"):
    state=torch.load(HERE/"checkpoints"/("masked" if mode=="masked" else "")/"model.pt",map_location="cpu",weights_only=True)
    if state["config"].get("image_side")!=SIDE or state["config"].get("patch_size")!=2:
        raise ValueError("Checkpoint pixel representation differs; train the current 48px patch model")
    signature=hashlib.sha256((HERE/"data"/"tokenizer.json").read_bytes()).hexdigest()
    if state["tokenizer_sha256"]!=signature:raise ValueError("Tokenizer changed; use the checkpoint's tokenizer or retrain")
    model=TextToPixels(**state["config"]).to(device).eval();model.load_state_dict(state["model"])
    return model,state


def main():
    parser=argparse.ArgumentParser(description="Describe a stick figure in ordinary English; the neural model predicts pixels.")
    parser.add_argument("prompt",help="For example: Draw a person bending forward and raising their left hand")
    parser.add_argument("--mode",choices=["masked","autoregressive"],default="autoregressive")
    parser.add_argument("--count",type=int,default=4)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--temperature",type=float,default=0.8,help="0 means greedy; positive values sample pixels")
    parser.add_argument("--guidance",type=float,default=2.,help="Text guidance scale; 1 disables classifier-free guidance")
    parser.add_argument("--output",type=Path,default=HERE/"samples"/"generated.png")
    args=parser.parse_args()
    if args.count<1 or args.temperature<0 or args.guidance<0:parser.error("Invalid count, temperature, or guidance")
    torch.set_num_threads(2);torch.manual_seed(args.seed)
    device="cuda" if torch.cuda.is_available() else "cpu"
    amp=device=="cuda" and torch.cuda.is_bf16_supported()
    tokenizer=load_tokenizer()
    try:ids=encode(args.prompt,tokenizer)
    except ValueError as error:parser.error(str(error))
    try:model,state=load_model(device,args.mode)
    except (ValueError,FileNotFoundError) as error:parser.error(str(error))
    captions=torch.tensor([ids]*args.count,device=device)
    empty=torch.tensor([[tokenizer.token_to_id("<eos>")]+[0]*(MAX_TEXT-1)],device=device)
    with torch.autocast(device,dtype=torch.bfloat16,enabled=amp):
        pixels=model.sample(captions,args.temperature,args.guidance,empty)
    save_grid(pixels,args.output,[args.prompt]*args.count,columns=min(4,args.count))
    print(f"Generated pixels using checkpoint step {state['step']}. Unspecified body parts may vary.")
    print(args.output.resolve())


if __name__=="__main__":main()
