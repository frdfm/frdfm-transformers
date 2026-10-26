"""Byte-level text tokenization only; no instruction interpretation."""
from pathlib import Path
from tokenizers import Tokenizer

HERE=Path(__file__).resolve().parent
MAX_TEXT=160
PAD=0
SIDE=24
PIXELS=SIDE*SIDE
PIXEL_BOS=2


def load_tokenizer():
    return Tokenizer.from_file(str(HERE/"data"/"tokenizer.json"))


def encode(text,tokenizer):
    if not text.strip():raise ValueError("Please describe the figure you want.")
    ids=tokenizer.encode(text).ids+[tokenizer.token_to_id("<eos>")]
    if len(ids)>MAX_TEXT:raise ValueError("Description is too long; please shorten it.")
    return ids+[PAD]*(MAX_TEXT-len(ids))
