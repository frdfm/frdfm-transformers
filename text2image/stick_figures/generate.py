import argparse
from pathlib import Path
import torch
from text2image.stick_figures.data import HERE, encode, encode_clauses, PARTS, SCALES
from text2image.stick_figures.model import model_class
from text2image.stick_figures.render import draw_figure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", help="Five semicolon-separated body instructions; see README")
    parser.add_argument("--architecture", choices=["pooled", "factorized"], default="factorized")
    parser.add_argument("--seed", type=int, default=7, help="Which trained model seed to load")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    encoder = encode_clauses if args.architecture == "factorized" else encode
    try:
        tokens = torch.tensor([encoder(args.prompt)])
    except ValueError as error:
        parser.error(str(error))
    suffix = Path() if args.architecture == "pooled" else Path(args.architecture)
    if args.output is None:
        args.output = HERE / "samples" / suffix / "generated.png"
    state = torch.load(HERE / "checkpoints" / suffix / f"seed-{args.seed}.pt", weights_only=True, map_location="cpu")
    if state.get("architecture", "pooled") != args.architecture:
        parser.error("Checkpoint architecture does not match the requested model")
    model = model_class(args.architecture)(**state["config"]).eval()
    model.load_state_dict(state["model"])
    with torch.inference_mode():
        controls = model(tokens)[0].numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    draw_figure(controls).save(args.output)
    print(dict(zip(PARTS, (controls*SCALES).round(2).tolist())))
    print(args.output.resolve())


if __name__ == "__main__":
    main()
