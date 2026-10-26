"""Offline synthetic raster generation. No renderer is used by model inference."""
import argparse
import hashlib
import json
import math
import shutil

import numpy as np
from PIL import Image, ImageDraw
from text2image.stick_pixels.schema import HERE, SIDE, PREFIX, encode


def raster(combo, rng, canonical=False):
    """Randomize anatomy/style while preserving discrete pose states and left/right."""
    from text2image.stick_figures.data import ANGLES
    def uniform(low, high):
        return (low + high) / 2 if canonical else rng.uniform(low, high)
    angles = np.array([ANGLES[i][state] for i,state in enumerate(combo)], dtype=float)
    if not canonical:
        # At most three degrees around the requested prototype; states stay separated.
        angles += rng.uniform(-3, 3, 5)
        angles = np.maximum(angles, 0)
    torso, left, right, left_knee, right_knee = np.deg2rad(angles)
    hip = np.array([0., 0.])
    torso_length = uniform(46, 66)
    direction = np.array([math.sin(torso), -math.cos(torso)])
    shoulder = hip + torso_length * direction
    radius = uniform(9, 15)
    neck = uniform(5, 11)
    head = shoulder + (neck + radius) * direction
    shoulder_half = uniform(5, 11)
    hip_half = uniform(9, 17)
    upper_arm, lower_arm = uniform(22, 32), uniform(21, 31)
    upper_leg, lower_leg = uniform(28, 42), uniform(26, 40)
    elbow_bend = uniform(0.02, 0.18)
    points = {"hip": hip, "shoulder": shoulder, "head": head}
    for label, sign, arm, knee_angle in [("left",-1,left,left_knee),("right",1,right,right_knee)]:
        start = shoulder + np.array([sign*shoulder_half, 0])
        elbow = start + upper_arm*np.array([sign*math.sin(arm), math.cos(arm)])
        wrist = elbow + lower_arm*np.array([sign*math.sin(arm+elbow_bend),math.cos(arm+elbow_bend)])
        knee = hip + np.array([sign*hip_half, upper_leg])
        ankle = knee + lower_leg*np.array([sign*math.sin(knee_angle),math.cos(knee_angle)])
        points.update({label+"_shoulder":start,label+"_elbow":elbow,label+"_wrist":wrist,
                       label+"_knee":knee,label+"_ankle":ankle})
    # Normalize the full drawing bounds, including the head, so no limb is cropped.
    all_points=np.stack([*points.values(),head+radius,head-radius])
    lower,upper=all_points.min(0),all_points.max(0)
    scale=uniform(0.84,1.04)*min(202/(upper[0]-lower[0]),208/(upper[1]-lower[1]))
    center=(lower+upper)/2
    offset=np.array([128+uniform(-9,9),128+uniform(-9,9)])
    points={name:(xy-center)*scale+offset for name,xy in points.items()}
    radius*=scale
    width=max(3,int(round(uniform(3.5,7.5)*scale)))
    image=Image.new("L",(256,256),0)
    pen=ImageDraw.Draw(image)
    def line(names):
        xy=[tuple(points[name]) for name in names]
        pen.line(xy,fill=255,width=width)
        for x,y in xy:
            r=width/2
            pen.ellipse((x-r,y-r,x+r,y+r),fill=255)
    line(["hip","shoulder"])
    line(["shoulder","head"])
    # Clear the center of the head so it remains an outlined circle.
    x,y=points["head"]
    pen.ellipse((x-radius,y-radius,x+radius,y+radius),fill=0,outline=255,width=width)
    for side in ("left","right"):
        line([side+"_shoulder",side+"_elbow",side+"_wrist"])
        line(["hip",side+"_knee",side+"_ankle"])
    pixels=(np.asarray(image.resize((SIDE,SIDE),Image.Resampling.LANCZOS))>=80).astype(np.uint8)
    return pixels.reshape(-1)


def archive_original(folder):
    manifest=folder/"manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()).get("format_version",1)==1:
        archive=folder/"translation_v1"
        archive.mkdir(exist_ok=True)
        for name in ("manifest.json","train.npz","val.npz","test.npz"):
            source=folder/name
            if source.exists() and not (archive/name).exists():
                shutil.copyfile(source,archive/name)
        return str(archive)


def build_dataset(folder, train_variants=256, eval_variants=64, seed=42):
    from text2image.stick_figures.data import split_combinations, describe
    if min(train_variants,eval_variants)<1:
        raise ValueError("Variant counts must be positive")
    folder.mkdir(parents=True,exist_ok=True)
    archive_original(folder)
    splits=split_combinations()
    counts,uniqueness,hashes={}, {}, {}
    prior_splits=set()
    for split_number,split in enumerate(("train","val","test")):
        images,captions,combos=[],[],[]
        variants=train_variants if split=="train" else eval_variants
        per_pose_unique=[]
        split_images=set()
        for combo_index,combo in enumerate(splits[split]):
            pose_images=set()
            for variant in range(variants):
                # Separate streams avoid RNG coupling between splits; duplicate rejection can resample.
                rng=np.random.default_rng(np.random.SeedSequence([seed,split_number,combo_index,variant]))
                for attempt in range(100):
                    image=raster(combo,rng,canonical=variant==0 and attempt==0)
                    key=np.packbits(image).tobytes()
                    if key not in prior_splits and key not in split_images:
                        break
                else:
                    raise ValueError(f"Could not generate a distinct raster for {split} {combo}")
                pose_images.add(key);split_images.add(key)
                # Caption shuffling has a separate RNG from image sampling.
                import random
                text_rng=random.Random(f"{seed}:{split}:{combo_index}:{variant}")
                captions.append(encode(describe(combo,text_rng)))
                images.append(image);combos.append(combo)
            per_pose_unique.append(len(pose_images))
        path=folder/f"{split}.npz"
        np.savez_compressed(path,pixels=np.stack(images),captions=np.array(captions,dtype=np.int64),
                            combos=np.array(combos,dtype=np.int64))
        hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        counts[split]=len(images)
        uniqueness[split]={"unique_rasters":len(split_images),"minimum_unique_per_pose":min(per_pose_unique),
                           "maximum_unique_per_pose":max(per_pose_unique)}
        prior_splits.update(split_images)
        print(f"{split}: {len(splits[split])} poses x {variants} variants = {len(images):,} records; "
              f"{len(split_images):,} unique rasters",flush=True)
    manifest={"format_version":2,"side":SIDE,"prefix":PREFIX,"levels":2,"seed":seed,
              "train_variants_per_combination":train_variants,"eval_variants_per_combination":eval_variants,
              "variants_per_split":{"train":train_variants,"val":eval_variants,"test":eval_variants},
              "combinations":{k:[list(c) for c in v] for k,v in splits.items()},"records":counts,
              "uniqueness":uniqueness,"files_sha256":hashes,
              "target":"binary pixels only; 0 background, 1 ink",
              "variation":"proportions, head radius, stroke, scale, position, elbow shape, <=3 degree pose jitter"}
    manifest["dataset_sha256"]=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    (folder/"manifest.json").write_text(json.dumps(manifest,indent=2))
    return manifest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--train-variants",type=int,default=256)
    parser.add_argument("--eval-variants",type=int,default=64)
    parser.add_argument("--seed",type=int,default=42)
    args=parser.parse_args()
    if min(args.train_variants,args.eval_variants)<1:parser.error("Variant counts must be positive")
    build_dataset(HERE/"data",args.train_variants,args.eval_variants,args.seed)


if __name__=="__main__":main()
