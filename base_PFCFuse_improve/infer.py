import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .base_PFCFuse import pfcfuse_fuse_gray
from .data import AGF_CONFIG, WLE_CONFIG, paired_files, prepare_pair, sha256


def main():
    parser = argparse.ArgumentParser(description="Fuse paired IR/VIS with the same AGF + WLE/MLE detail pipeline as main.py")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("agf", "raw"), default="agf")
    parser.add_argument("--detail-fusion", choices=("wle", "mle"), default="wle",
                        help="Detail fusion used with --mode agf; must match the checkpoint")
    parser.add_argument("--baseline", action="store_true", help="Disable learned residual adapters")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.limit < 0:
        parser.error("limit must be nonnegative")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    metadata = checkpoint.get("metadata")
    if metadata is not None and metadata["mode"] != args.mode:
        parser.error("Inference mode must match training mode")
    if metadata is not None and metadata.get("detail_fusion") != (args.detail_fusion if args.mode == "agf" else "none"):
        parser.error("Inference detail fusion must match training mode")
    if metadata is not None and metadata.get("detail_fusion_config") != (WLE_CONFIG if args.mode == "agf" and args.detail_fusion == "wle" else None):
        parser.error("Inference detail fusion configuration must match training mode")
    pairs = paired_files(args.source)
    if args.limit:
        pairs = pairs[:args.limit]
    args.output.mkdir(parents=True, exist_ok=False)
    for index, (name, visible, infrared) in enumerate(pairs):
        arrays = prepare_pair(visible, infrared, args.mode, args.detail_fusion)
        fused_base = pfcfuse_fuse_gray(arrays["base_infrared"], arrays["base_visible"],
                                      ckpt_path=args.checkpoint, device=args.device, baseline=args.baseline)
        fused = np.clip(fused_base + arrays["detail"], 0, 1)
        if not cv2.imwrite(str(args.output / f"{name}.png"), (fused * 255).astype(np.uint8)):
            raise OSError(f"Failed to write {name}")
        print(f"[{index + 1}/{len(pairs)}] {name}", flush=True)
    record = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha256(args.checkpoint),
                  source=str(args.source.resolve()), mode=args.mode, detail_fusion=args.detail_fusion if args.mode == "agf" else "none",
                  detail_fusion_config=WLE_CONFIG if args.mode == "agf" and args.detail_fusion == "wle" else None,
                  baseline=args.baseline, agf=AGF_CONFIG if args.mode == "agf" else None, images=len(pairs))
    (args.output / "inference.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()