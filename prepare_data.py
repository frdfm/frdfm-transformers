import json
import os

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

MAX_LEN = 60
TRAIN_PAIRS = 996_000  # opus-100 en-fr has ~1M pairs total; this takes essentially all of it
VAL_PAIRS = 4_000
VOCAB_SIZE = 16_000

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

# ---- BPE: train one shared EN+FR subword vocab, PAD=0 / EOS=1 reserved ----

bpe = models.BPE(unk_token=None)
tokenizer = Tokenizer(bpe)
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=["<pad>", "<eos>"],
    show_progress=False,
)  # BpeTrainer starts from the alphabet implied by the corpus automatically
tokenizer.train_from_iterator((en + " " + fr for en, fr in pairs), trainer=trainer)
tokenizer.save(os.path.join(DATA_DIR, "tokenizer.json"))

pad_id = tokenizer.token_to_id("<pad>")
eos_id = tokenizer.token_to_id("<eos>")


def encode_row(text):
    row = np.zeros(MAX_LEN, dtype=np.uint16)
    ids = tokenizer.encode(text).ids[: MAX_LEN - 1] + [eos_id]
    row[: len(ids)] = ids
    return row


def dump_src_then_tgt(name, group):
    src_rows = np.stack([encode_row(en) for en, _ in group])
    tgt_rows = np.stack([encode_row(fr) for _, fr in group])
    src_rows.tofile(os.path.join(DATA_DIR, f"{name}_src.bin"))
    tgt_rows.tofile(os.path.join(DATA_DIR, f"{name}_tgt.bin"))


dump_src_then_tgt("train", train_pairs)
dump_src_then_tgt("val", val_pairs)

print(f"train pairs {len(train_pairs):,}")
print(f"val pairs {len(val_pairs):,}")
print(f"vocab {tokenizer.get_vocab_size():,}")
print(f"saved {DATA_DIR}")
