"""Fixed ordinary-English requests for visual review; only neural pixels are drawn."""
import json,argparse
import torch
from text2image.text2stick.tokenization import HERE,MAX_TEXT,load_tokenizer,encode
from text2image.text2stick.generate import load_model,save_grid

PROMPTS=[
    "Draw a person standing upright with both hands raised above their head.",
    "Draw someone bending forward deeply, raising their left hand and keeping their right arm down.",
    "Draw someone bending forward deeply, raising their right hand and keeping their left arm down.",
    "Draw a person standing straight with their left arm stretched sideways and their right knee bent.",
    "Draw a person bending over with both knees bent and both arms down.",
    "Draw a stick figure standing upright with both arms out to the sides and both legs straight.",
]


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=["autoregressive","masked"],default="autoregressive")
    args=parser.parse_args()
    torch.set_num_threads(2);torch.manual_seed(42)
    device="cuda" if torch.cuda.is_available() else "cpu"
    amp=device=="cuda" and torch.cuda.is_bf16_supported()
    tokenizer=load_tokenizer();model,state=load_model(device,args.mode)
    captions=torch.tensor([encode(p,tokenizer) for p in PROMPTS],device=device)
    empty=torch.tensor([[tokenizer.token_to_id("<eos>")]+[0]*(MAX_TEXT-1)],device=device)
    with torch.autocast(device,dtype=torch.bfloat16,enabled=amp):
        pixels=model.sample(captions,temperature=0.7,guidance=2.,empty_text=empty)
    save_grid(pixels,HERE/"samples"/args.mode/"plain-english-demo.png",PROMPTS,columns=3)
    (HERE/"samples"/"demo-prompts.json").write_text(json.dumps(PROMPTS,indent=2))
    print(HERE/"samples"/args.mode/"plain-english-demo.png")


if __name__=="__main__":main()
