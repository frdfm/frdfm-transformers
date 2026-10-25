import os
import sys

import torch
from tokenizers import Tokenizer

from en2fr.model import Model
from en2fr.pipeline import MAX_LEN, PAD, EOS, BOS, check_data, encode_source, tokenizer_digest

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
CKPT_FILE = os.path.join(HERE, "checkpoints", "model.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    check_data(DATA_DIR)
    tokenizer = Tokenizer.from_file(os.path.join(DATA_DIR, "tokenizer.json"))
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "the weather is nice today"
    src = torch.tensor([encode_source(tokenizer, prompt)], dtype=torch.long, device=DEVICE)
    state = torch.load(CKPT_FILE, map_location=DEVICE, weights_only=True)
    if (state.get("format_version") != 2
            or state.get("tokenizer_sha256") != tokenizer_digest(DATA_DIR)):
        raise ValueError("Incompatible checkpoint: retrain with python -m en2fr.train after preparing data.")
    model = Model().to(DEVICE)
    model.load_state_dict(state["model"])
    model.eval()
    with torch.inference_mode():
        memory = model.encoder(src)
        outputs = [BOS]
        for _ in range(MAX_LEN):
            tgt = torch.tensor([outputs], dtype=torch.long, device=DEVICE)
            logits = model.decoder(tgt, memory, src == PAD)[:, -1].clone()
            logits[:, [PAD, BOS]] = -torch.inf
            next_token = logits.argmax(dim=-1).item()
            if next_token == EOS:
                break
            outputs.append(next_token)
    print(tokenizer.decode(outputs[1:]))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        sys.exit(str(error))
