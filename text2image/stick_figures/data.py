"""Exhaustive discrete pose space; split combinations before generating captions."""
import itertools
import json
from pathlib import Path
import random
import re

import numpy as np

HERE = Path(__file__).resolve().parent
PARTS = ("torso", "left arm", "right arm", "left knee", "right knee")
STATES = (("upright", "slightly bent", "deeply bent"), ("down", "out", "raised"),
          ("down", "out", "raised"), ("straight", "bent"), ("straight", "bent"))
ANGLES = ((0, 25, 50), (0, 80, 160), (0, 80, 160), (0, 65), (0, 65))
SCALES = np.array([50, 160, 160, 65, 65], dtype=np.float32)
COMBINATIONS = list(itertools.product(range(3), range(3), range(3), range(2), range(2)))
WORDS = sorted(set(re.findall(r"[a-z]+|;", " ; ".join(f"{part} {state}" for part, states in zip(PARTS, STATES) for state in states))))
VOCAB = {word: i + 2 for i, word in enumerate(WORDS)}
PAD, CLS, MAX_TEXT = 0, 1, 32


def encode(text):
    words = re.findall(r"[a-z]+|;", text.lower())
    if not words or len(words) + 1 > MAX_TEXT:
        raise ValueError("Use a nonempty pose description of at most 31 word/separator tokens")
    unknown = set(words) - VOCAB.keys()
    if unknown:
        raise ValueError(f"Unknown words: {sorted(unknown)}. Use the README prompt syntax.")
    ids = [CLS] + [VOCAB[w] for w in words]
    return ids + [PAD] * (MAX_TEXT - len(ids))


CLAUSES, CLAUSE_TEXT = 5, 6


def encode_clauses(text):
    """Syntax-only segmentation, no body-part/state interpretation."""
    clauses = text.lower().split(";")
    if len(clauses) != CLAUSES:
        raise ValueError("Provide exactly five semicolon-separated clauses, one per body part")
    rows = []
    for clause in clauses:
        ids = [CLS] + encode(clause)[1:]
        ids = [i for i in ids if i != PAD]
        if len(ids) > CLAUSE_TEXT:
            raise ValueError("Each clause must contain at most five words")
        rows.append(ids + [PAD] * (CLAUSE_TEXT - len(ids)))
    return rows


def clause_supervision(records):
    """Training-only labels from our synthetic annotation schema, never used by generation."""
    parts, values = [], []
    dictionary = {f"{part} {state}": i for i, (part, states) in enumerate(zip(PARTS, STATES)) for state in states}
    for record in records:
        identities = [dictionary[c.strip()] for c in record["caption"].split(";")]
        parts.append(identities)
        values.append([record["target"][i] for i in identities])
    return parts, values


def describe(combo, rng=None):
    clauses = [f"{part} {states[state]}" for part, states, state in zip(PARTS, STATES, combo)]
    if rng is not None:
        rng.shuffle(clauses)
    return " ; ".join(clauses)


def target(combo):
    return [ANGLES[i][state] / float(SCALES[i]) for i, state in enumerate(combo)]


def split_combinations(seed=42):
    # Reserve entire attribute pairings, not merely complete five-part tuples.
    # Test: deep torso bend AND raised left arm, with every other limb combination.
    # Validation: slight torso bend AND raised right arm, with every other combination.
    split = {"train": [], "val": [], "test": []}
    for combo in COMBINATIONS:
        name = "test" if combo[0] == 2 and combo[1] == 2 else (
               "val" if combo[0] == 1 and combo[2] == 2 else "train")
        split[name].append(combo)
    for i, states in enumerate(STATES):
        if {c[i] for c in split["train"]} != set(range(len(states))):
            raise ValueError("An individual primitive is missing from training")
    return split


def make_dataset(seed=42, variants=24):
    split = split_combinations(seed)
    records = {}
    for split_index, (name, combos) in enumerate(split.items()):
        rng = random.Random(seed + 100 * split_index)
        records[name] = [{"combo": list(combo), "caption": describe(combo, rng), "target": target(combo)}
                         for combo in combos for _ in range(variants)]
    return {"split_seed": seed, "variants_per_combination": variants,
            "combinations": {name: [list(c) for c in combos] for name, combos in split.items()},
            "records": records}


def main():
    data = make_dataset()
    folder = HERE / "data"
    folder.mkdir(exist_ok=True)
    (folder / "dataset.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    for name in ("train", "val", "test"):
        print(f"{name}: {len(data['combinations'][name])} combinations, {len(data['records'][name])} captions")


if __name__ == "__main__":
    main()
