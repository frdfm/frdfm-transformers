"""Evaluate language conditioning and directly generated pixel images; no geometry inference."""
import argparse,json
import numpy as np
import torch
from text2image.stick_colors.tokenization import HERE,MAX_TEXT,PIXEL_BOS,SIDE,PATCH,encode,load_tokenizer
from text2image.stick_colors.generate import load_model,save_grid
from text2image.stick_colors.train import load,decoder_inputs,per_image_loss
from text2image.stick_colors.metrics import diagnose



@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=["masked","autoregressive"],default="autoregressive")
    parser.add_argument("--splits",nargs="+",default=["word_test","pose_test","combined_test"])
    parser.add_argument("--count",type=int,default=12)
    parser.add_argument("--guidance",type=float,default=2.)
    parser.add_argument("--temperature",type=float,default=0.)
    args=parser.parse_args()
    if args.count<1 or args.temperature<0 or args.guidance<0:parser.error("Invalid evaluation arguments")
    torch.set_num_threads(2);torch.manual_seed(42)
    device="cuda" if torch.cuda.is_available() else "cpu"
    amp=device=="cuda" and torch.cuda.is_bf16_supported()
    model,state=load_model(device,args.mode)
    tokenizer=load_tokenizer()
    empty=torch.tensor([[tokenizer.token_to_id("<eos>")]+[0]*(MAX_TEXT-1)],device=device)
    reports=dict(mode=args.mode,image_side=SIDE,patch_size=PATCH,pixel_score_kind="autoregressive_patch_nll_per_pixel" if args.mode=="autoregressive" else "fully_masked_patch_cross_entropy_per_pixel",step=state["step"],temperature=args.temperature,guidance=args.guidance,splits={})
    for split in args.splits:
        if split not in ("pose_val","word_val","word_test","pose_test","combined_test"):parser.error("Unknown split")
        data=load(split);total=0.;gap=0.;count=0
        # Caption mismatch uses a different full pose, and is a diagnostic only.
        permutation=torch.roll(torch.arange(len(data["pixels"])),shifts=64 if "pose" in split or "combined" in split else 16)
        for batch in torch.arange(len(data["pixels"])).split(32):
            text=data["text"][batch].to(device);pixels=data["tokens"][batch].to(device)
            wrong=data["text"][permutation[batch]].to(device)
            with torch.autocast(device,dtype=torch.bfloat16,enabled=amp):
                scores=model(torch.cat([text,wrong]),torch.cat([torch.full_like(pixels,PIXEL_BOS) if args.mode=="masked" else decoder_inputs(pixels)]*2))
            good,bad=scores.chunk(2)
            good_loss=per_image_loss(good,pixels);bad_loss=per_image_loss(bad,pixels)
            total+=good_loss.sum().item();gap+=(bad_loss-good_loss).sum().item();count+=len(batch)
        # First record of each distinct pose, before any model outputs are inspected.
        selected=[];seen=set()
        for i,combo in enumerate(data["combos"].tolist()):
            if tuple(combo) not in seen:
                seen.add(tuple(combo));selected.append(i)
            if len(selected)>=args.count:break
        indices=torch.tensor(selected)
        texts=json.loads((HERE/"data"/f"{split}-captions.json").read_text())
        labels=[texts[i] for i in selected]
        with torch.autocast(device,dtype=torch.bfloat16,enabled=amp):
            predicted=model.sample(data["text"][indices].to(device),args.temperature,args.guidance,empty).cpu()
        reference=data["pixels"][indices]
        shuffle=torch.randperm(len(predicted),generator=torch.Generator().manual_seed(99))
        reports["splits"][split]=dict(pixel_loss=total/count,mismatched_caption_nll_increase=gap/count,
                   generated_count=len(indices),generated=diagnose(predicted.numpy(),reference.numpy()),targets=diagnose(reference.numpy()),
                   shuffled_alignment=diagnose(predicted[shuffle].numpy(),reference.numpy()))
        panels=torch.stack([reference,predicted],1).reshape(-1,reference.shape[1])
        panel_labels=[prefix+label for label in labels for prefix in ("REFERENCE: ","GENERATED: ")]
        save_grid(panels,HERE/"samples"/args.mode/f"{split}-comparison.png",panel_labels,columns=4)
        save_grid(panels[:2],HERE/"samples"/args.mode/f"{split}-example.png",panel_labels[:2],columns=2)
        folder=HERE/"results"/("masked" if args.mode=="masked" else "");folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/f"{split}-generated.npz",pixels=predicted.numpy(),reference=reference.numpy(),indices=indices.numpy())
        print(split,json.dumps(reports["splits"][split]),flush=True)
    suffix="validation" if all("val" in s for s in args.splits) else "evaluation"
    (folder/f"{suffix}-t{args.temperature:g}.json").write_text(json.dumps(reports,indent=2))


if __name__=="__main__":main()
