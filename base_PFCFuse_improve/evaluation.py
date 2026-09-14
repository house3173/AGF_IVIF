import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader

from .data import CachedPairs
from .fusion_model import FORMAT, load_model
from .objective import FusionObjective
from .utils.Evaluator import Evaluator


def image_metrics(fused, visible, infrared, unclipped):
    fused = np.clip(fused, 0, 1)
    fused_u8 = (fused * 255).astype(np.uint8)
    visible_u8 = np.rint(visible * 255).astype(np.uint8)
    infrared_u8 = np.rint(infrared * 255).astype(np.uint8)
    output, source_vi, source_ir = [image.astype(np.float64) for image in (fused_u8, visible_u8, infrared_u8)]
    qabf = (Evaluator.Qabf(output, source_vi, source_ir) if np.any(source_vi) or np.any(source_ir)
            else float(not np.any(output)))
    def correlation(first, second):
        first, second = first - first.mean(), second - second.mean()
        denominator = np.linalg.norm(first) * np.linalg.norm(second)
        return float((first * second).sum() / denominator) if denominator > 1e-12 else 0.0

    metrics = {
        "Qabf": float(qabf),
        "SSIM": float((structural_similarity(output, source_vi, data_range=255)
                       + structural_similarity(output, source_ir, data_range=255)) / 2),
        "SCD": correlation(source_vi, output - source_ir) + correlation(source_ir, output - source_vi),
        "MI_bits": float(Evaluator.MI(fused_u8, visible_u8, infrared_u8) / np.log(2)),
        "EN": float(Evaluator.EN(output)),
        "SF": float(Evaluator.SF(output)),
        "clipping": float(np.mean((unclipped < 0) | (unclipped > 1))),
    }
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError("Non-finite evaluation metric; inspect the input images")
    return metrics


def to_device(batch, device):
    return {name: value.to(device) if isinstance(value, torch.Tensor) else value
            for name, value in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device, output_dir=None):
    model.eval()
    objective = FusionObjective().to(device)
    rows = []
    for batch in loader:
        batch = to_device(batch, device)
        if batch["visible"].shape[0] != 1:
            raise ValueError("Full-image evaluation requires batch_size=1")
        features = model.encode(batch["base_visible"], batch["base_infrared"])
        reference = model.decode(batch["base_visible"], features, baseline=True)
        candidate = model.decode(batch["base_visible"], features)
        row = {"id": batch["id"][0]}
        visible = batch["visible"][0, 0].cpu().numpy()
        infrared = batch["infrared"][0, 0].cpu().numpy()
        for name, base in (("original", reference), ("improved", candidate)):
            unclipped = (base + batch["detail"])[0, 0].cpu().numpy()
            fused = unclipped.clip(0, 1)
            metrics = image_metrics(fused, visible, infrared, unclipped)
            metrics["loss"] = float(objective(base, batch, reference))
            row.update({f"{name}_{key}": value for key, value in metrics.items()})
            if output_dir is not None:
                folder = Path(output_dir) / name
                folder.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(folder / f"{row['id']}.png"), (fused * 255).astype(np.uint8)):
                    raise OSError(f"Failed to write {row['id']}")
        rows.append(row)
    if not rows:
        raise ValueError("Evaluation requires at least one image")
    means = {key: float(np.mean([row[key] for row in rows])) for key in rows[0] if key != "id"}
    return means, rows


def eligible(means, min_gain=1e-4):
    return (
        means["improved_Qabf"] > means["original_Qabf"] + min_gain
        and means["improved_SSIM"] >= means["original_SSIM"] - 0.001
        and means["improved_SCD"] >= means["original_SCD"] - 0.01
        and means["improved_clipping"] <= means["original_clipping"] + 0.001
        and means["improved_loss"] < means["original_loss"]
    )


def paired_intervals(rows, samples=2000, seed=3407):
    generator = np.random.default_rng(seed)
    result = {}
    for key in rows[0]:
        if not key.startswith("original_"):
            continue
        metric = key[len("original_"):]
        differences = np.array([row[f"improved_{metric}"] - row[key] for row in rows])
        estimates = differences[generator.integers(0, len(rows), size=(samples, len(rows)))].mean(axis=1)
        result[metric] = dict(delta=float(differences.mean()),
                              ci95=np.quantile(estimates, [0.025, 0.975]).tolist(),
                              fraction_increased=float(np.mean(differences > 0)))
    return result


def main():
    parser = argparse.ArgumentParser(description="Paired original/improved full-image evaluation")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    dataset = CachedPairs(args.cache, args.split)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if payload.get("format") != FORMAT:
        parser.error("Use a checkpoint produced by train_residual, including its data provenance")
    if payload["metadata"]["mode"] != dataset.manifest["mode"]:
        parser.error("Checkpoint mode does not match cache mode")
    if payload["metadata"].get("detail_fusion") != dataset.manifest.get("detail_fusion"):
        parser.error("Checkpoint detail fusion does not match cache detail fusion")
    if payload["metadata"].get("detail_fusion_config") != dataset.manifest.get("detail_fusion_config"):
        parser.error("Checkpoint detail fusion configuration does not match cache")
    if args.split == "test":
        used = set(payload["metadata"]["used_digests"])
        if any(item["digest"] in used for item in dataset.records):
            parser.error("Test overlaps the training/validation pairs used by this checkpoint")
    args.output.mkdir(parents=True, exist_ok=False)
    model = load_model(args.checkpoint, device=args.device)
    started = time.perf_counter()
    means, rows = evaluate(model, DataLoader(dataset, batch_size=1), args.device, args.output)
    elapsed = time.perf_counter() - started
    intervals = paired_intervals(rows)
    supported = len(rows) >= 20 and eligible(means) and intervals["Qabf"]["ci95"][0] > 0
    report = dict(means=means, paired=intervals, images=len(rows), elapsed_seconds=elapsed,
                  checkpoint=str(args.checkpoint.resolve()), split=args.split, mode=dataset.manifest["mode"],
                  evidence="supported_on_this_split" if supported else "not_established",
                  caveat="Image bootstrap assumes independent images; not a guarantee for new scenes or datasets.")
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    with (args.output / "per_image.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()