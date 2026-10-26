"""Append new training variations; freeze tokenizer, original rows, and all holdouts."""
import argparse,hashlib,json,random,shutil
from collections import Counter
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
from text2image.stick_colors.tokenization import HERE,SIDE,encode
from text2image.stick_colors.renderer import raster,is_connected
from text2image.text2stick.captions import caption


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def append_variants(arrays,captions,tokenizer,target,seen,seed=2027,progress=None):
    """Keep the complete old prefix. Use only combinations already present in train."""
    counts=Counter(map(tuple,arrays['combos'].tolist()))
    if len(set(counts.values()))!=1:raise ValueError('Training poses must have equal variant counts')
    current=next(iter(counts.values()))
    if target<=current:raise ValueError(f'Target must exceed the current {current} variants per pose')
    additions={key:[] for key in arrays};new_captions=[]
    for combo_number,combo in enumerate(sorted(counts)):
        for j in range(current,target):
            rng=np.random.default_rng(np.random.SeedSequence([seed,combo_number,j]))
            for _ in range(200):
                pixels=raster(combo,rng,side=SIDE)
                key=hashlib.sha256(pixels.tobytes()).digest()
                if key not in seen and set(np.unique(pixels))==set(range(7)) and is_connected(pixels,SIDE):
                    seen.add(key);break
            else:raise ValueError('Unable to generate a distinct connected target')
            words=random.Random(f'42:train:{combo_number}:{j}')
            partial=words.random()<0.3
            text=caption(combo,words,None,partial)
            full=caption(combo,random.Random(f'full:train:{combo_number}:{j}'),None,False)
            values=dict(pixels=pixels,text=encode(text,tokenizer),full_text=encode(full,tokenizer),combo=combo,partial=partial)
            for name in arrays:additions[name].append(values['combo' if name=='combos' else name])
            new_captions.append(text)
        if progress and ((combo_number+1)%12==0 or combo_number+1==len(counts)):
            progress(f'Expanded {combo_number+1}/{len(counts)} poses to {target} variants',flush=True)
    expanded={name:np.concatenate([array,np.asarray(additions[name],dtype=array.dtype)],axis=0) for name,array in arrays.items()}
    return expanded,captions+new_captions,current


def expand(folder,target=896,seed=2027):
    folder=Path(folder)
    manifest=json.loads((folder/'manifest.json').read_text())
    for name,expected in manifest['files_sha256'].items():
        if sha(folder/name)!=expected:raise ValueError(f'Data checksum mismatch: {name}')
    if sha(folder/'tokenizer.json')!=manifest['tokenizer_sha256']:raise ValueError('Tokenizer checksum mismatch')
    with np.load(folder/'train.npz') as data:arrays={key:data[key] for key in data.files}
    captions=json.loads((folder/'train-captions.json').read_text())
    counts=Counter(map(tuple,arrays['combos'].tolist()))
    if len(set(counts.values()))!=1 or target<=next(iter(counts.values())):
        raise ValueError('Expansion target must exceed the current per-pose count')
    frozen=[file for file in folder.iterdir() if file.is_file() and file.name not in ('train.npz','train-captions.json','manifest.json')]
    frozen_hashes={file.name:sha(file) for file in frozen}
    archive=folder/'archives'/manifest['dataset_sha256'][:16];archive.mkdir(parents=True,exist_ok=True)
    for file in folder.iterdir():
        if file.is_file() and not (archive/file.name).exists():shutil.copy2(file,archive/file.name)
    seen=set()
    for name in manifest['files_sha256']:
        with np.load(folder/name) as data:
            seen.update(hashlib.sha256(row.tobytes()).digest() for row in data['pixels'])
    tokenizer=Tokenizer.from_file(str(folder/'tokenizer.json'))
    expanded,texts,previous=append_variants(arrays,captions,tokenizer,target,seen,seed,print)
    staging=folder/'.expanded';staging.mkdir(exist_ok=True)
    np.savez_compressed(staging/'train.npz',**expanded)
    (staging/'train-captions.json').write_text(json.dumps(texts,indent=2))
    for name,expected in frozen_hashes.items():
        if sha(folder/name)!=expected:raise ValueError(f'Frozen file changed: {name}')
    old_count=len(captions)
    for name,original in arrays.items():np.testing.assert_array_equal(expanded[name][:old_count],original)
    new=dict(manifest)
    new.pop('dataset_sha256',None)
    new.update(format_version=2,parent_dataset_sha256=manifest['dataset_sha256'],
        expansion=dict(seed=seed,previous_variants_per_pose=previous,variants_per_pose=target,
                       retained_train_records=old_count,training_poses=len(counts),
                       retained_train_npz_sha256=sha(folder/'train.npz'),archive=str(archive.relative_to(folder))),
        frozen_files_sha256=frozen_hashes)
    new['records']=dict(manifest['records'],train=len(texts))
    new['files_sha256']=dict(manifest['files_sha256'],**{'train.npz':sha(staging/'train.npz')})
    new['dataset_sha256']=hashlib.sha256(json.dumps(new,sort_keys=True).encode()).hexdigest()
    (staging/'manifest.json').write_text(json.dumps(new,indent=2))
    for name in ('train.npz','train-captions.json','manifest.json'):(staging/name).replace(folder/name)
    print(f'Training: {old_count:,} -> {len(texts):,}; tokenizer and all holdouts unchanged',flush=True)
    return new


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variants-per-pose',type=int,default=896)
    parser.add_argument('--seed',type=int,default=2027)
    args=parser.parse_args()
    try:expand(HERE/'data',args.variants_per_pose,args.seed)
    except ValueError as error:parser.error(str(error))


if __name__=='__main__':main()
