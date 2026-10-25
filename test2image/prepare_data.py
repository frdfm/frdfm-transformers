"""Download checked Fashion-MNIST IDX files; no torchvision dependency."""
import argparse
import gzip
import hashlib
from pathlib import Path
import struct
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F

from test2image.model import SIDE, LEVELS

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
BASE = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
RESOURCES = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}


def fetch(name):
    path = DATA / name
    if not path.exists() or hashlib.md5(path.read_bytes()).hexdigest() != RESOURCES[name]:
        temporary = path.with_suffix(".part")
        print(f"Downloading {name}", flush=True)
        urllib.request.urlretrieve(BASE + name, temporary)
        if hashlib.md5(temporary.read_bytes()).hexdigest() != RESOURCES[name]:
            raise ValueError(f"Checksum failed for {name}")
        temporary.replace(path)
    return gzip.decompress(path.read_bytes())


def read_split(name, side=28):
    images, labels = fetch(name + "-images-idx3-ubyte.gz"), fetch(name + "-labels-idx1-ubyte.gz")
    magic, count, height, width = struct.unpack(">IIII", images[:16])
    label_magic, label_count = struct.unpack(">II", labels[:8])
    if (magic, label_magic, height, width) != (2051, 2049, 28, 28) or count != label_count:
        raise ValueError("Invalid Fashion-MNIST headers")
    array = np.frombuffer(images, dtype=np.uint8, offset=16).copy().reshape(count, 28, 28)
    labels = np.frombuffer(labels, dtype=np.uint8, offset=8).copy()
    x = torch.from_numpy(array).float().unsqueeze(1) / 255
    x = F.adaptive_avg_pool2d(x, (side, side)).squeeze(1)
    pixels = (x * (LEVELS - 1)).round().to(torch.uint8).reshape(count, side * side)
    return pixels.numpy(), labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, choices=[14, 28], default=28)
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    destination = DATA if args.size == 14 else DATA / str(args.size)
    destination.mkdir(parents=True, exist_ok=True)
    x, y = read_split("train", args.size)
    order = np.random.default_rng(42).permutation(len(x))
    # Validation comes from training data; official test split remains untouched.
    for name, pick in [("val", order[:5000]), ("train", order[5000:])]:
        np.savez_compressed(destination / f"{name}.npz", pixels=x[pick], labels=y[pick])
    x, y = read_split("t10k", args.size)
    np.savez_compressed(destination / "test.npz", pixels=x, labels=y)
    print(f"Saved {args.size}x{args.size}: 55,000 training / 5,000 validation / 10,000 test images to {destination}")


if __name__ == "__main__":
    main()
