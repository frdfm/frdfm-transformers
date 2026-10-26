"""Build ordinary-English captions paired with binary pixels; no pose regression targets."""
import hashlib
import json
import random
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer,models,pre_tokenizers,decoders,trainers
from text2image.text2stick.captions import caption
from text2image.text2stick.tokenization import HERE,encode,SIDE,MAX_TEXT


def build_records():
    from text2image.text2stick.renderer import raster,is_connected
    from text2image.stick_figures.data import split_combinations
    combinations=split_combinations()
    manifest={"combinations":combinations,"variants_per_split":{"train":256,"val":64,"test":64}}
    variants=manifest["variants_per_split"]
    source_images={}
    seen=set()
    for split_number,name in enumerate(("train","val","test")):
        images=[]
        for combo_number,combo in enumerate(combinations[name]):
            for j in range(variants[name]):
                rng=np.random.default_rng(np.random.SeedSequence([2026,split_number,combo_number,j]))
                for attempt in range(200):
                    pixels=raster(combo,rng,side=SIDE)
                    key=np.packbits(pixels).tobytes()
                    if key not in seen and is_connected(pixels,SIDE):
                        seen.add(key);images.append(pixels);break
                else:raise ValueError("Unable to create a distinct raster")
        source_images[name]=np.stack(images)
        print(f"Rendered {name}: {len(images)} connected-style targets at {SIDE}px",flush=True)
    manifest["dataset_sha256"]=hashlib.sha256(b"".join(source_images[n].tobytes() for n in ("train","val","test"))).hexdigest()
    groups={}
    variants=manifest["variants_per_split"]
    source_pixels=source_images["train"]
    for split,offset,count,family in [("train",0,224,None),("word_val",224,16,3),("word_test",240,16,4)]:
        rows=[]
        for combo_number,combo in enumerate(manifest["combinations"]["train"]):
            for j in range(offset,offset+count):
                rng=random.Random(f"42:{split}:{combo_number}:{j}")
                partial=split=="train" and rng.random()<0.3
                text=caption(combo,rng,family,partial)
                full=caption(combo,random.Random(f"full:{split}:{combo_number}:{j}"),family,False)
                rows.append(dict(text=text,full_text=full,partial=partial,combo=combo,
                                 pixels=source_pixels[combo_number*variants["train"]+j].copy()))
        groups[split]=rows
    for split,source_split,family in [("pose_val","val",None),("pose_test","test",None),("combined_test","test",4)]:
        rows=[]
        combos=np.repeat(np.array(combinations[source_split]),variants[source_split],axis=0)
        for i,(image,combo) in enumerate(zip(source_images[source_split],combos)):
            text=caption(combo,random.Random(f"42:{split}:{i}"),family,False)
            rows.append(dict(text=text,full_text=text,partial=False,combo=combo.tolist(),pixels=image.copy()))
        groups[split]=rows
    return groups,manifest["dataset_sha256"]


def main():
    folder=HERE/"data";folder.mkdir(exist_ok=True)
    groups,source_hash=build_records()
    tokenizer=Tokenizer(models.BPE())
    tokenizer.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder=decoders.ByteLevel()
    trainer=trainers.BpeTrainer(vocab_size=1024,special_tokens=["<pad>","<eos>"],
                               initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),show_progress=False)
    tokenizer.train_from_iterator((text for row in groups["train"] for text in (row["text"],row["full_text"])),trainer)
    tokenizer.save(str(folder/"tokenizer.json"))
    files={};counts={}
    for split,records in groups.items():
        tokens=np.array([encode(r["text"],tokenizer) for r in records],dtype=np.int32)
        full=np.array([encode(r["full_text"],tokenizer) for r in records],dtype=np.int32)
        path=folder/f"{split}.npz"
        np.savez_compressed(path,text=tokens,full_text=full,pixels=np.stack([r["pixels"] for r in records]),
                            combos=np.array([r["combo"] for r in records],dtype=np.int16),
                            partial=np.array([r["partial"] for r in records]))
        (folder/f"{split}-captions.json").write_text(json.dumps([r["text"] for r in records],indent=2))
        files[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        counts[split]=len(records)
        print(split,len(records),"maximum text length",int((tokens!=0).sum(1).max()),flush=True)
    manifest=dict(format_version=2,renderer="connected_skeleton_48px_v2",representation="plain_english_to_binary_pixels",side=SIDE,max_text=MAX_TEXT,
                  tokenizer_sha256=hashlib.sha256((folder/"tokenizer.json").read_bytes()).hexdigest(),
                  source_dataset_sha256=source_hash,records=counts,files_sha256=files,
                  split_policy="pose pairings held out; phrase families 0-2 train,3 wording val,4 wording test")
    manifest["dataset_sha256"]=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    (folder/"manifest.json").write_text(json.dumps(manifest,indent=2))


if __name__=="__main__":main()
