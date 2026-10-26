import argparse,json,time,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from text2image.stick_colors.tokenization import HERE,PIXEL_BOS,SIDE,MAX_TEXT,PATCH,PIXEL_CLASSES,load_tokenizer,encode
from text2image.stick_colors.model import ColorPixels
from text2image.stick_colors.pixel_tokens import pack_pixels


def load(split):
    with np.load(HERE/"data"/f"{split}.npz") as d:
        data={key:torch.from_numpy(d[key]) for key in d.files}
    for key in ("text","full_text"):data[key]=data[key].long()
    data["tokens"]=torch.cat([pack_pixels(chunk) for chunk in data["pixels"].split(1024)])
    return data


def validate_resume_data(state,manifest,accept_expanded=False,folder=None):
    folder=HERE/"data" if folder is None else folder
    if state["tokenizer_sha256"]!=manifest["tokenizer_sha256"]:
        raise ValueError("Checkpoint/tokenizer mismatch")
    if state["dataset_sha256"]==manifest["dataset_sha256"]:return
    if not accept_expanded:
        raise ValueError("Checkpoint/data mismatch; for a verified append-only expansion use --accept-expanded-data")
    if manifest.get("parent_dataset_sha256")!=state["dataset_sha256"]:
        raise ValueError("Expanded data must directly descend from the checkpoint dataset")
    expansion=manifest.get("expansion",{})
    archive=(folder/expansion["archive"]).resolve()
    if not archive.is_relative_to(folder.resolve()):raise ValueError("Archive must stay inside the data folder")
    def signature(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    parent=json.loads((archive/"manifest.json").read_text())
    parent_signature=parent.pop("dataset_sha256")
    if parent_signature!=state["dataset_sha256"] or hashlib.sha256(json.dumps(parent,sort_keys=True).encode()).hexdigest()!=parent_signature:
        raise ValueError("Archived parent manifest does not match checkpoint data")
    if expansion["retained_train_npz_sha256"]!=parent["files_sha256"]["train.npz"] or expansion["retained_train_records"]!=parent["records"]["train"]:
        raise ValueError("Original training provenance changed")
    required=(set(parent["files_sha256"])-{"train.npz"})|{"tokenizer.json"}
    required.update(file.name for file in archive.glob("*-captions.json") if file.name!="train-captions.json")
    if not required.issubset(manifest["frozen_files_sha256"]):raise ValueError("Frozen evaluation files are missing")
    if signature(folder/"tokenizer.json")!=state["tokenizer_sha256"]:raise ValueError("Tokenizer changed")
    for name,expected in manifest["frozen_files_sha256"].items():
        if signature(folder/name)!=expected or signature(archive/name)!=expected:
            raise ValueError(f"Frozen evaluation/tokenizer file changed: {name}")
    for name,expected in manifest["files_sha256"].items():
        if signature(folder/name)!=expected:raise ValueError(f"Expanded dataset checksum mismatch: {name}")
    if signature(archive/"train.npz")!=expansion["retained_train_npz_sha256"]:
        raise ValueError("Original training archive changed")
    with np.load(archive/"train.npz") as old,np.load(folder/"train.npz") as new:
        count=expansion["retained_train_records"]
        for key in old.files:
            if not np.array_equal(old[key],new[key][:count]):raise ValueError("Original training rows changed")
        old_combos=set(map(tuple,old["combos"].tolist()))
        new_combos=set(map(tuple,new["combos"].tolist()))
        if old_combos!=new_combos:raise ValueError("Expansion changed training pose combinations")
    print("Accepted verified training-only expansion; preserving optimizer and absolute learning-rate step.",flush=True)


def decoder_inputs(pixels):
    return torch.cat([torch.full_like(pixels[:,:1],PIXEL_BOS),pixels[:,:-1]],1)


def per_image_loss(logits,pixels,ink_weight=1.,mask=None):
    losses=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),pixels.reshape(-1),reduction="none").reshape_as(pixels)
    patch_bits=PATCH*PATCH
    ink=sum(((pixels//(7**bit))%7!=0).float() for bit in range(patch_bits))/patch_bits
    weights=1+(ink_weight-1)*ink
    if mask is not None:weights=weights*mask
    return (losses*weights).sum(1)/weights.sum(1).clamp_min(1)/patch_bits


@torch.inference_mode()
def validation(model,data,device,amp):
    model.eval();total=0.;count=0
    # Four fixed style variants per semantic pose, to keep periodic selection inexpensive.
    stride=16 if len(data["pixels"])==1344 else 64
    indices=torch.tensor([i for start in range(0,len(data["pixels"]),stride) for i in range(start,min(start+4,len(data["pixels"])))])
    for batch in indices.split(32):
        text,pixels=data["text"][batch].to(device),data["tokens"][batch].to(device)
        with torch.autocast(device,dtype=torch.bfloat16,enabled=amp):
            scores=model(text,decoder_inputs(pixels))
        total+=per_image_loss(scores,pixels).sum().item();count+=len(batch)
    return total/count


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=["autoregressive"],default="autoregressive")
    parser.add_argument("--initialize",type=str,help="Optional locally trained checkpoint for weight initialization")
    parser.add_argument("--steps",type=int,default=4000)
    parser.add_argument("--batch-size",type=int,default=16)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--accept-expanded-data",action="store_true",help="Resume only a verified append-only training expansion with frozen holdouts/tokenizer")
    parser.add_argument("--seed",type=int,default=7)
    args=parser.parse_args()
    if args.accept_expanded_data and not args.resume:parser.error("--accept-expanded-data requires --resume")
    if min(args.steps,args.batch_size)<1:parser.error("Steps and batch size must be positive")
    torch.set_num_threads(2);torch.manual_seed(args.seed)
    device="cuda" if torch.cuda.is_available() else "cpu"
    amp=device=="cuda" and torch.cuda.is_bf16_supported()
    train=load("train");validations=[load(s) for s in ("pose_val","word_val")]
    tokenizer=load_tokenizer();manifest=json.loads((HERE/"data"/"manifest.json").read_text())
    empty=torch.tensor([tokenizer.token_to_id("<eos>")]+[0]*(MAX_TEXT-1))
    model=ColorPixels(tokenizer.get_vocab_size(),mode=args.mode).to(device)
    if args.initialize and not args.resume:
        initial=torch.load(args.initialize,map_location="cpu",weights_only=True)
        if initial["tokenizer_sha256"]!=manifest["tokenizer_sha256"]:raise ValueError("Initialization tokenizer mismatch")
        if initial["config"]!=model.config:raise ValueError("Initialization model/color representation mismatch")
        model.load_state_dict(initial["model"])
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.0005,weight_decay=0.01)
    suffix="masked" if args.mode=="masked" else ""
    checkpoint=HERE/"checkpoints"/suffix;checkpoint.mkdir(parents=True,exist_ok=True)
    results=HERE/"results"/suffix;results.mkdir(parents=True,exist_ok=True)
    start,best,history=0,float("inf"),[]
    if args.resume:
        state=torch.load(checkpoint/"latest.pt",map_location="cpu",weights_only=True)
        validate_resume_data(state,manifest,args.accept_expanded_data)
        model.load_state_dict(state["model"]);optimizer.load_state_dict(state["optimizer"])
        start,best=state["step"],state["best_validation"]
        torch.set_rng_state(state["cpu_rng"])
        if amp and "cuda_rng" in state:torch.cuda.set_rng_state_all(state["cuda_rng"])
        if (results/"history.json").exists():history=json.loads((results/"history.json").read_text())
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters; {device}; bf16={amp}; English -> pixels",flush=True)
    started=time.time()
    for step in range(start+1,start+args.steps+1):
        model.train()
        pick=torch.randint(len(train["pixels"]),(args.batch_size,))
        negative=torch.randint(len(train["pixels"]),(args.batch_size,))
        same=(train["combos"][pick]==train["combos"][negative]).all(1)
        while same.any():
            negative[same]=torch.randint(len(train["pixels"]),(int(same.sum()),))
            same=(train["combos"][pick]==train["combos"][negative]).all(1)
        text=train["text"][pick].clone()
        blank=torch.rand(args.batch_size)<0.1
        text[blank]=empty
        pixels=train["tokens"][pick].to(device)
        negative_text=train["full_text"][negative]
        # Only fully specified positive descriptions enter mismatch ranking.
        rank_valid=(~train["partial"][pick].bool() & ~blank).to(device)
        lr=0.0005*min(step/200,(200/step)**0.5)
        optimizer.param_groups[0]["lr"]=lr
        mask=None
        inputs=decoder_inputs(pixels)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device,dtype=torch.bfloat16,enabled=amp):
            scores=model(torch.cat([text,negative_text],0).to(device),torch.cat([inputs,inputs],0))
            positive,wrong=scores.chunk(2)
            positive_nll=per_image_loss(positive,pixels,mask=mask)
            wrong_nll=per_image_loss(wrong,pixels,mask=mask)
            ranking=F.relu(0.025+positive_nll-wrong_nll)
            ranking=(ranking*rank_valid).sum()/rank_valid.sum().clamp_min(1)
            loss=per_image_loss(positive,pixels,ink_weight=2.,mask=mask).mean()+0.5*ranking
        if not torch.isfinite(loss):raise RuntimeError("Nonfinite training loss")
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
        if step%250==0 or step==start+args.steps:
            values=[validation(model,d,device,amp) for d in validations]
            selection=sum(values)/len(values)
            improved=selection<best;best=min(best,selection)
            row=dict(step=step,train_loss=loss.item(),positive_nll=positive_nll.mean().item(),
                     ranking_loss=ranking.item(),pose_val=values[0],word_val=values[1],
                     validation=selection,lr=lr,run_seconds=time.time()-started)
            history.append(row);print(json.dumps(row),flush=True)
            state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),config=model.config,
                       step=step,best_validation=best,dataset_sha256=manifest["dataset_sha256"],
                       tokenizer_sha256=manifest["tokenizer_sha256"],cpu_rng=torch.get_rng_state(),
                       representation="palette_pixels",image_side=SIDE,seed=args.seed,mode=args.mode)
            if device=="cuda":state["cuda_rng"]=torch.cuda.get_rng_state_all()
            torch.save(state,checkpoint/"latest.tmp");(checkpoint/"latest.tmp").replace(checkpoint/"latest.pt")
            if improved:
                torch.save(state,checkpoint/"best.tmp");(checkpoint/"best.tmp").replace(checkpoint/"model.pt")
            (results/"history.json").write_text(json.dumps(history,indent=2))
    if device=="cuda":print(f"Peak allocated GPU memory {torch.cuda.max_memory_allocated()/2**30:.2f} GiB",flush=True)


if __name__=="__main__":main()
