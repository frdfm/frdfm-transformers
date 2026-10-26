"""Text/pixel vocabulary only. No geometry or pose decoding."""
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
SIDE, PREFIX, LEVELS = 24, 32, 2
PIXELS = SIDE * SIDE
# Pixel IDs 0=white background and 1=black ink. Caption IDs cannot collide.
PAD, START = 2, 3
WORDS = sorted("torso upright slightly deeply bent left right arm down out raised knee straight ;".split())
VOCAB = {word: i + 4 for i, word in enumerate(WORDS)}
INPUT_VOCAB = len(VOCAB) + 4


def encode(text):
    words = re.findall(r"[a-z]+|;", text.lower())
    unknown = set(words) - VOCAB.keys()
    if not words or unknown or len(words) + 1 > PREFIX:
        raise ValueError(f"Use the supported pose clauses; unknown words: {sorted(unknown)}")
    ids = [START] + [VOCAB[w] for w in words]
    return ids + [PAD] * (PREFIX - len(ids))


def dataset_signature():
    import json, hashlib
    manifest=json.loads((HERE/"data"/"manifest.json").read_text())
    for name, expected in manifest.get("files_sha256",{}).items():
        if hashlib.sha256((HERE/"data"/name).read_bytes()).hexdigest()!=expected:
            raise ValueError(f"Dataset checksum mismatch: {name}; prepare the data again")
    return manifest.get("dataset_sha256","legacy-v1"),manifest.get("format_version",1)
