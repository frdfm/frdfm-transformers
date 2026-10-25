import json
import os

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from pipeline import MAX_LEN, VOCAB_SIZE, PAD, EOS, BOS, tokenizer_digest

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

TRAIN_PAIRS = 996_000  # opus-100 en-fr has ~1M pairs total; this takes essentially all of it
VAL_PAIRS = 4_000

os.makedirs(DATA_DIR, exist_ok=True)

stream = load_dataset("Helsinki-NLP/opus-100", "en-fr", split="train", streaming=True)

pairs = []
for example in stream:
    en = example["translation"]["en"].strip()
    fr = example["translation"]["fr"].strip()
    if not en or not fr:
        continue
    pairs.append((en, fr))
    if len(pairs) >= VAL_PAIRS + TRAIN_PAIRS:
        break

val_pairs = pairs[:VAL_PAIRS]
train_pairs = pairs[VAL_PAIRS:]

# Shared byte-level BPE: PAD=0, EOS=1, BOS=2, UNK=3.

bpe = models.BPE(unk_token="<unk>")
tokenizer = Tokenizer(bpe)
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=["<pad>", "<eos>", "<bos>", "<unk>"],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    show_progress=False,
)
tokenizer.train_from_iterator((en + " " + fr for en, fr in train_pairs), trainer=trainer)
tokenizer.save(os.path.join(DATA_DIR, "tokenizer.json"))

pad_id = tokenizer.token_to_id("<pad>")
eos_id = tokenizer.token_to_id("<eos>")


def encode_row(text):
    row = np.zeros(MAX_LEN, dtype=np.uint16)
    ids = tokenizer.encode(text).ids
    if len(ids) >= MAX_LEN:
        return None
    ids = ids + [eos_id]
    row[: len(ids)] = ids
    return row


def dump_src_then_tgt(name, group):
    src_rows, tgt_rows = [], []
    for en, fr in group:
        src, tgt = encode_row(en), encode_row(fr)
        if src is None or tgt is None:
            continue
        src_rows.append(src)
        tgt_rows.append(tgt)
    if not src_rows:
        raise ValueError(f"No {name} pairs fit within {MAX_LEN} tokens.")
    np.stack(src_rows).tofile(os.path.join(DATA_DIR, f"{name}_src.bin"))
    np.stack(tgt_rows).tofile(os.path.join(DATA_DIR, f"{name}_tgt.bin"))
    print(f"{name}: kept {len(src_rows):,}, filtered {len(group) - len(src_rows):,} long pairs")


dump_src_then_tgt("train", train_pairs)
dump_src_then_tgt("val", val_pairs)

print(f"train pairs {len(train_pairs):,}")
print(f"val pairs {len(val_pairs):,}")
print(f"vocab {tokenizer.get_vocab_size():,}")
print(f"saved {DATA_DIR}")

assert (pad_id, eos_id, tokenizer.token_to_id("<bos>")) == (PAD, EOS, BOS)
with open(os.path.join(DATA_DIR, "metadata.json"), "w") as file:
    json.dump({"format_version": 2, "max_len": MAX_LEN, "vocab_size": VOCAB_SIZE,
               "tokenizer_sha256": tokenizer_digest(DATA_DIR)}, file, indent=2)
