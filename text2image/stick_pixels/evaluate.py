"""Evaluate actual generated pixels; never convert predicted poses into a drawing."""
import json
import numpy as np
import torch
from text2image.stick_pixels.schema import HERE,VOCAB,START,PAD,dataset_signature
from text2image.stick_pixels.model import PixelTransformer
from text2image.stick_pixels.train import load,evaluate
from text2image.stick_pixels.generate import save_grid


def overlap(prediction,expected):
    p,y=prediction.bool(),expected.bool()
    intersection=(p&y).sum(1).float();union=(p|y).sum(1).float()
    return {"mean_ink_iou":float((intersection/union.clamp_min(1)).mean()),
            "pixel_accuracy":float((p==y).float().mean()),
            "blank_image_count":int((p.sum(1)==0).sum()),
            "mean_ink_pixels":float(p.sum(1).float().mean())}


def main():
    torch.set_num_threads(2);torch.manual_seed(42)
    device="cuda" if torch.cuda.is_available() else "cpu"
    state=torch.load(HERE/"checkpoints"/"model.pt",map_location=device,weights_only=True)
    assert state["representation"]=="binary_pixels"
    signature,version=dataset_signature()
    if state.get("dataset_sha256","legacy-v1")!=signature:
        raise SystemExit("Checkpoint was trained on different data; train a fresh model on the new dataset before evaluating.")
    manifest=json.loads((HERE/"data"/"manifest.json").read_text())
    model=PixelTransformer(**state["config"]).to(device).eval();model.load_state_dict(state["model"])
    report={"checkpoint_step":state["step"],"seed":state["seed"],"representation":"generated binary pixels"}
    reverse={v:k for k,v in VOCAB.items()}
    for split in ("train","val","test"):
        captions,pixels=load(split)
        loss=evaluate(model,captions,pixels,device)
        variants=manifest.get("variants_per_split",{}).get(split,manifest.get("variants_per_combination",24))
        indices=torch.arange(0,len(pixels),variants)[:12]
        text=captions[indices];reference=pixels[indices]
        labels=[" ".join(reverse[int(i)] for i in row if int(i) not in (PAD,START)).replace(" ; "," ; ") for row in text]
        prediction=model.sample(text.to(device),temperature=0).cpu()
        report[split]={"teacher_forced_loss":loss,"generation_count":len(indices),
                       "greedy":overlap(prediction,reference),
                       "all_background_baseline":overlap(torch.zeros_like(reference),reference),
                       "shuffled_caption_alignment":overlap(prediction[torch.randperm(len(prediction),generator=torch.Generator().manual_seed(99))],reference)}
        folder=HERE/"results";folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/f"{split}-generated-pixels.npz",reference=reference.numpy(),
                            prediction=prediction.numpy(),captions=text.numpy())
        if split=="test":
            torch.manual_seed(42)
            sampled=model.sample(text.to(device),temperature=0.8).cpu()
            report[split]["sampled_temperature_0.8"]=overlap(sampled,reference)
            save_grid(sampled,HERE/"samples"/"test-sampled.png",labels,columns=4)
        panels=torch.stack([reference,prediction],1).reshape(-1,reference.shape[1])
        captions_for_grid=[prefix+label for label in labels for prefix in ("REFERENCE ; ","GENERATED ; ")]
        save_grid(panels,HERE/"samples"/f"{split}-comparison.png",captions_for_grid,columns=4)
        if split=="test":
            save_grid(panels[:2],HERE/"samples"/"test-example.png",captions_for_grid[:2],columns=2)
        print(split,json.dumps(report[split]),flush=True)
    folder=HERE/"results";folder.mkdir(exist_ok=True)
    (folder/"evaluation.json").write_text(json.dumps(report,indent=2))


if __name__=="__main__":main()
