import argparse
from pathlib import Path
import torch
from text2image.stick_figures.data import HERE, encode, PARTS, SCALES
from text2image.stick_figures.model import PoseTransformer
from text2image.stick_figures.render import draw_figure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", help="Five semicolon-separated body instructions; see README")
    parser.add_argument("--seed", type=int, default=7, help="Which trained model seed to load")
    parser.add_argument("--output", type=Path, default=HERE / "samples" / "generated.png")
    args = parser.parse_args()
    tokens = torch.tensor([encode(args.prompt)])
    state = torch.load(HERE / "checkpoints" / f"seed-{args.seed}.pt", weights_only=True, map_location="cpu")
    model = PoseTransformer(**state["config"]).eval()
    model.load_state_dict(state["model"])
    with torch.inference_mode():
        controls = model(tokens)[0].numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    draw_figure(controls).save(args.output)
    print(dict(zip(PARTS, (controls*SCALES).round(2).tolist())))
    print(args.output.resolve())


if __name__ == "__main__":
    main()
