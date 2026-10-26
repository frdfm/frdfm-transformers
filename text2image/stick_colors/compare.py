"""Show real targets next to unfiltered neural samples for the same English prompt."""
import argparse,json,textwrap
import numpy as np
import torch
from PIL import Image,ImageDraw
from text2image.stick_colors.tokenization import HERE,SIDE,MAX_TEXT,load_tokenizer,encode
from text2image.stick_colors.generate import load_model
from text2image.stick_colors.palette import RGB,NAMES
from text2image.stick_colors.metrics import diagnose


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--split',choices=['train','word_test','pose_test','combined_test'],default='train')
    parser.add_argument('--count',type=int,default=4)
    parser.add_argument('--seed',type=int,default=123)
    parser.add_argument('--temperature',type=float,default=0.8)
    args=parser.parse_args()
    if args.count<1 or args.temperature<0:parser.error('Invalid count or temperature')
    torch.set_num_threads(2);torch.manual_seed(args.seed)
    with np.load(HERE/'data'/f'{args.split}.npz') as data:
        eligible=np.flatnonzero(~data['partial'].astype(bool))
        if args.count>len(eligible):parser.error('Too many examples')
        indices=np.random.default_rng(42).choice(eligible,args.count,replace=False)
        reference=data['pixels'][indices]
    captions=json.loads((HERE/'data'/f'{args.split}-captions.json').read_text())
    prompts=[captions[int(i)] for i in indices]
    device='cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer=load_tokenizer();model,state=load_model(device)
    text=torch.tensor([encode(p,tokenizer) for p in prompts for _ in range(2)],device=device)
    empty=torch.tensor([[tokenizer.token_to_id('<eos>')]+[0]*(MAX_TEXT-1)],device=device)
    with torch.autocast(device,dtype=torch.bfloat16,enabled=device=='cuda' and torch.cuda.is_bf16_supported()):
        generated=model.sample(text,args.temperature,2.,empty).cpu().numpy().reshape(args.count,2,SIDE,SIDE)
    tile=240;gap=12;width=768
    lines=[textwrap.wrap(f'{args.split} #{int(i)}: {p}',width=104) for i,p in zip(indices,prompts)]
    heights=[len(row)*16+42+tile+14 for row in lines]
    canvas=Image.new('RGB',(width,46+sum(heights)),'#eeeeee');draw=ImageDraw.Draw(canvas)
    for color,name in enumerate(NAMES[1:],1):
        x=12+(color-1)*126
        draw.rectangle((x,10,x+14,24),fill=tuple(RGB[color]))
        draw.text((x+19,11),name,fill='black')
    y=46
    for row,caption_lines in enumerate(lines):
        for j,line in enumerate(caption_lines):draw.text((12,y+8+j*16),line,fill='black')
        top=y+len(caption_lines)*16+16
        for col,(label,pixels) in enumerate(zip(('ACTUAL TARGET','GENERATED SAMPLE 1','GENERATED SAMPLE 2'),[reference[row].reshape(SIDE,SIDE),generated[row,0],generated[row,1]])):
            x=gap+col*(tile+gap)
            draw.text((x,top),label,fill='black')
            canvas.paste(Image.fromarray(RGB[pixels]).resize((tile,tile),Image.Resampling.NEAREST),(x,top+20))
        y+=heights[row]
    folder=HERE/'samples';folder.mkdir(exist_ok=True)
    path=folder/f'{args.split}-comparison.png';canvas.save(path)
    flattened=generated.reshape(-1,SIDE*SIDE)
    repeated=np.repeat(reference,2,axis=0)
    metadata=dict(split=args.split,indices=indices.tolist(),prompts=prompts,checkpoint_step=state['step'],
                  seed=args.seed,temperature=args.temperature,guidance=2.,generated=diagnose(flattened,repeated),targets=diagnose(reference))
    (folder/f'{args.split}-comparison.json').write_text(json.dumps(metadata,indent=2))
    np.savez_compressed(folder/f'{args.split}-comparison.npz',reference=reference,generated=generated,indices=indices)
    print(path.resolve())
    print(json.dumps(metadata['generated'],indent=2))


if __name__=='__main__':main()
