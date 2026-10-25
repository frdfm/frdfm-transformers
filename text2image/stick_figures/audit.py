"""Evaluate all 120 clause permutations of each held-out combination; no training."""
import argparse
import itertools
import json
import numpy as np
import torch
from text2image.stick_figures.data import HERE, describe, encode_clauses, target, SCALES
from text2image.stick_figures.model import FactorizedPoseTransformer
from text2image.stick_figures.train import predict, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[7,11,23])
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = json.loads((HERE / "data" / "dataset.json").read_text())
    records, canonical, targets = [], [], []
    for combo in data["combinations"]["test"]:
        clauses = describe(combo).split(" ; ")
        canonical.append(encode_clauses(describe(combo)))
        for order in itertools.permutations(clauses):
            records.append(encode_clauses(" ; ".join(order)))
            targets.append(target(combo))
    tokens, expected = torch.tensor(records), torch.tensor(targets)
    reports = []
    for seed in args.seeds:
        state = torch.load(HERE / "checkpoints" / "factorized" / f"seed-{seed}.pt", map_location="cpu", weights_only=True)
        model = FactorizedPoseTransformer(**state["config"]).to(device).eval()
        model.load_state_dict(state["model"])
        predictions = predict(model, tokens, device)
        reference = predict(model, torch.tensor(canonical), device).repeat_interleave(120, dim=0)
        report = {"seed":seed,"poses":len(canonical),"orderings_per_pose":120,"captions":len(records),
                  "metrics":metrics(predictions, expected),
                  "max_order_drift_degrees":float(((predictions-reference).abs()*torch.tensor(SCALES)).max()),
                  "max_target_error_degrees":float(((predictions-expected).abs()*torch.tensor(SCALES)).max())}
        reports.append(report)
        print(json.dumps(report, indent=2), flush=True)
    (HERE / "results" / "factorized" / "permutation_audit.json").write_text(json.dumps(reports,indent=2))


if __name__ == "__main__":
    main()
