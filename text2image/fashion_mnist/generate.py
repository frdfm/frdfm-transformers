import argparse
from pathlib import Path
import torch
from PIL import Image, ImageDraw
from text2image.fashion_mnist.model import ImageTransformer, encode_prompt, CLASSES, SIDE, LEVELS

HERE = Path(__file__).resolve().parent


def save_grid(pixels, path, columns=8, labels=None, scale=12):
    images = pixels.detach().cpu().numpy()
    rows = (len(images) + columns - 1) // columns
    cell = images.shape[-1] * scale
    grid = Image.new("L", (columns * cell, rows * (cell + 20)), 255)
    draw = ImageDraw.Draw(grid)
    for i, array in enumerate(images):
        # Invert for dark sketches on white; the model trains on original grayscale.
        image = Image.fromarray((255 - array * (255 / (LEVELS - 1))).round().astype("uint8"))
        x, y = (i % columns) * cell, (i // columns) * (cell + 20)
        grid.paste(image.resize((cell, cell), Image.Resampling.NEAREST), (x, y))
        if labels:
            draw.text((x + 3, y + cell + 2), labels[i], fill=0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, choices=[14, 28], default=28)
    parser.add_argument("prompt", nargs="?", default="draw a bag")
    parser.add_argument("--all", action="store_true", help="Four variations of all ten categories")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.count < 1 or args.temperature <= 0:
        parser.error("Count and temperature must be positive")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    suffix = Path() if args.size == 14 else Path(str(args.size))
    if args.output is None:
        args.output = HERE / "samples" / suffix / "generated.png"
    state = torch.load(HERE / "checkpoints" / suffix / "model.pt", map_location=device, weights_only=True)
    model = ImageTransformer(**state["config"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    prompts = [f"draw a {c}" for c in CLASSES for _ in range(4)] if args.all else [args.prompt] * args.count
    captions = torch.tensor([encode_prompt(p) for p in prompts], device=device)
    pixels = model.sample(captions, args.temperature)
    save_grid(pixels, args.output, columns=4 if args.all else min(8, args.count), labels=prompts)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
