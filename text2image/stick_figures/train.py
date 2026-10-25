"""Train against combination-held-out validation; evaluate test only after selection."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from text2image.stick_figures.data import HERE, PARTS, ANGLES, SCALES, encode, encode_clauses, clause_supervision, target, describe, make_dataset
from text2image.stick_figures.model import model_class
from text2image.stick_figures.render import save_comparison


def tensors(records, architecture="pooled"):
    encoder = encode_clauses if architecture == "factorized" else encode
    return (torch.tensor([encoder(r["caption"]) for r in records]),
            torch.tensor([r["target"] for r in records], dtype=torch.float32))


@torch.inference_mode()
def predict(model, tokens, device):
    model.eval()
    return torch.cat([model(batch.to(device)).cpu() for batch in tokens.split(128)])


def metrics(predictions, expected):
    predictions, expected = predictions.numpy(), expected.numpy()
    correct = []
    for part, values in enumerate(ANGLES):
        choices = np.array(values) / SCALES[part]
        predicted_class = abs(predictions[:, part, None] - choices).argmin(1)
        expected_class = abs(expected[:, part, None] - choices).argmin(1)
        correct.append(predicted_class == expected_class)
    correct = np.stack(correct, 1)
    errors = abs(predictions - expected) * SCALES
    return {"mse": float(np.mean((predictions - expected)**2)),
            "component_accuracy": {name: float(correct[:, i].mean()) for i, name in enumerate(PARTS)},
            "all_five_accuracy": float(correct.all(1).mean()),
            "mean_absolute_degrees": {name: float(errors[:, i].mean()) for i, name in enumerate(PARTS)},
            "all_five_within_10_degrees": float((errors <= 10).all(1).mean())}


def run(seed, args, data, device):
    torch.manual_seed(seed)
    train_x, train_y = tensors(data["records"]["train"], args.architecture)
    val_x, val_y = tensors(data["records"]["val"], args.architecture)
    model = model_class(args.architecture)().to(device)
    identities, clause_values = map(torch.tensor, clause_supervision(data["records"]["train"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    suffix = Path() if args.architecture == "pooled" else Path(args.architecture)
    output = HERE / "checkpoints" / suffix / f"seed-{seed}.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    best, best_step = float("inf"), 0
    history = []
    started = time.time()
    for step in range(1, 1 if args.evaluate_only else args.steps + 1):
        model.train()
        batch = torch.randint(len(train_x), (64,))
        lr = 0.001 * min(step / 100, (100 / step)**0.5)
        optimizer.param_groups[0]["lr"] = lr
        if args.architecture == "factorized":
            prediction, part_logits, clause_angles = model.details(train_x[batch].to(device))
            routing_loss = F.cross_entropy(part_logits.reshape(-1, 5), identities[batch].to(device).reshape(-1))
            local_loss = F.mse_loss(clause_angles, clause_values[batch].to(device))
            loss = F.mse_loss(prediction, train_y[batch].to(device)) + local_loss + 0.25 * routing_loss
        else:
            prediction = model(train_x[batch].to(device))
            loss = F.mse_loss(prediction, train_y[batch].to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0 or step == args.steps:
            validation = F.mse_loss(predict(model, val_x, device), val_y).item()
            history.append({"step": step, "training_mse": loss.item(), "validation_mse": validation})
            if validation < best:
                best, best_step = validation, step
                torch.save({"model": model.state_dict(), "config": model.config, "step": step,
                            "seed": seed, "architecture": args.architecture, "split_seed": data["split_seed"], "validation_mse": best}, output)
            if step % 500 == 0 or step == args.steps:
                print(f"seed {seed} step {step} train {loss.item():.5f} val {validation:.5f} seconds {time.time()-started:.0f}", flush=True)
    # Selection is finished before test records are evaluated.
    state = torch.load(output, weights_only=True, map_location=device)
    model.load_state_dict(state["model"])
    best_step = state["step"]
    if args.evaluate_only:
        history_file = HERE / "results" / suffix / "validation.json"
        if history_file.exists():
            history = next((r["history"] for r in json.loads(history_file.read_text()) if r["seed"] == seed), [])
    if args.validation_only:
        return {"seed": seed, "architecture": args.architecture, "selected_step": best_step,
                "parameters": sum(p.numel() for p in model.parameters()),
                "val": metrics(predict(model, val_x, device), val_y), "history": history}
    test_x, test_y = tensors(data["records"]["test"], args.architecture)
    test_prediction = predict(model, test_x, device)
    report = {"seed": seed, "architecture": args.architecture, "parameters": sum(p.numel() for p in model.parameters()),
              "selected_step": best_step, "seconds": time.time()-started,
              "train": metrics(predict(model, train_x, device), train_y),
              "val": metrics(predict(model, val_x, device), val_y),
              "test": metrics(test_prediction, test_y),
              "text_ignored_mean_baseline": metrics(train_y.mean(0).expand_as(test_y), test_y),
              "shuffled_test_caption_control": metrics(test_prediction[torch.randperm(len(test_prediction), generator=torch.Generator().manual_seed(99))], test_y),
              "history": history}
    if args.architecture == "factorized":
        model.eval()
        with torch.inference_mode():
            outputs = [model.details(batch.to(device)) for batch in test_x.split(128)]
            routing = torch.cat([o[1].cpu() for o in outputs]).argmax(-1)
        actual_parts, _ = clause_supervision(data["records"]["test"])
        report["test_clause_routing_accuracy"] = float((routing == torch.tensor(actual_parts)).float().mean())
    # Fixed combination IDs, not cherry-picked good predictions. Canonical caption order.
    combos = data["combinations"]["test"][:6]
    captions = [describe(c) for c in combos]
    encoder = encode_clauses if args.architecture == "factorized" else encode
    predicted = predict(model, torch.tensor([encoder(c) for c in captions]), device)
    report["preview"] = [{"caption": c, "expected_degrees": (np.array(target(combo))*SCALES).tolist(),
                          "predicted_degrees": (p.numpy()*SCALES).tolist()} for c, combo, p in zip(captions, combos, predicted)]
    save_comparison([(c, target(combo), p.numpy()) for c, combo, p in zip(captions, combos, predicted)],
                    HERE / "samples" / suffix / f"held-out-seed-{seed}.png")
    save_comparison([(captions[0], target(combos[0]), predicted[0].numpy())],
                    HERE / "samples" / suffix / f"example-seed-{seed}.png")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=["pooled", "factorized"], default="factorized")
    parser.add_argument("--evaluate-only", action="store_true", help="Evaluate existing validation-selected checkpoints without retraining")
    parser.add_argument("--validation-only", action="store_true", help="Do not evaluate or preview test examples")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 23])
    args = parser.parse_args()
    if args.evaluate_only and args.validation_only:
        parser.error("Choose validation-only or evaluate-only, not both")
    if args.steps < 1:
        parser.error("Steps must be positive")
    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = HERE / "data" / "dataset.json"
    if not data_path.exists():
        raise FileNotFoundError("Run python -m text2image.stick_figures.data first")
    data = json.loads(data_path.read_text(encoding="utf-8"))
    if data != make_dataset(data["split_seed"], data["variants_per_combination"]):
        raise ValueError("Dataset is incompatible with the current generator")
    print(f"device {device}; train/val/test combinations: " + str({k: len(v) for k,v in data["combinations"].items()}), flush=True)
    results = [run(seed, args, data, device) for seed in args.seeds]
    folder = HERE / "results" / (args.architecture if args.architecture != "pooled" else "")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / ("validation.json" if args.validation_only else "metrics.json")).write_text(json.dumps(results, indent=2), encoding="utf-8")
    for result in results:
        if args.validation_only:
            print(json.dumps({"seed": result["seed"], "selected_step": result["selected_step"], "val": result["val"]}, indent=2))
            continue
        print(json.dumps({"seed": result["seed"], "selected_step": result["selected_step"], "test": result["test"],
                          "mean_baseline_all_five": result["text_ignored_mean_baseline"]["all_five_accuracy"],
                          "shuffled_caption_all_five": result["shuffled_test_caption_control"]["all_five_accuracy"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
