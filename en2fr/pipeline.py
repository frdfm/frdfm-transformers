"""Shared token IDs, source encoding, and artifact compatibility checks."""
import hashlib
import json
from pathlib import Path

MAX_LEN = 60
VOCAB_SIZE = 16_000
PAD, EOS, BOS = 0, 1, 2


def tokenizer_digest(data_dir):
    return hashlib.sha256((Path(data_dir) / "tokenizer.json").read_bytes()).hexdigest()


def check_data(data_dir):
    path = Path(data_dir) / "metadata.json"
    if not path.exists():
        raise ValueError("Old data format: run python -m en2fr.prepare_data, then retrain with python -m en2fr.train.")
    metadata = json.loads(path.read_text())
    if (metadata.get("format_version") != 2
            or metadata.get("max_len") != MAX_LEN
            or metadata.get("vocab_size") != VOCAB_SIZE
            or metadata.get("tokenizer_sha256") != tokenizer_digest(data_dir)):
        raise ValueError("Incompatible data/tokenizer: run python -m en2fr.prepare_data, then python -m en2fr.train.")


def encode_source(tokenizer, text):
    if not text.strip():
        raise ValueError("Enter a non-empty English sentence.")
    ids = tokenizer.encode(text).ids
    if len(ids) >= MAX_LEN:
        raise ValueError(f"Sentence is too long: use at most {MAX_LEN - 1} tokens.")
    return ids + [EOS]
