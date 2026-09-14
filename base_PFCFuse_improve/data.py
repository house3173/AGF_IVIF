import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


AGF_CONFIG = dict(sigma_spatial=5, sigma_range=0.05, iterations=4, median_ksize=3, numba=True)
WLE_CONFIG = dict(um=5, pad_mode="reflect")
CHANNELS = ("visible", "infrared", "base_visible", "base_infrared", "detail")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired_files(root):
    def collect(folder):
        result = {}
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                if path.stem in result:
                    raise ValueError(f"Duplicate image ID: {path.stem}")
                result[path.stem] = path
        return result

    visible = collect(Path(root) / "vi")
    infrared = collect(Path(root) / "ir")
    if not visible or visible.keys() != infrared.keys():
        raise ValueError("Nonempty IR/VIS folders must contain exactly matching image IDs")
    return [(name, visible[name], infrared[name]) for name in sorted(visible)]


def prepare_pair(visible_path, infrared_path, mode, detail_fusion="wle"):
    images = []
    for path in (visible_path, infrared_path):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or min(image.shape) < 8:
            raise ValueError(f"Unreadable or too small: {path}")
        images.append(image.astype(np.float32) / 255.0)
    visible, infrared = images
    if visible.shape != infrared.shape:
        raise ValueError(f"Unaligned shapes: {visible_path}, {infrared_path}")
    if mode == "agf":
        from decomposition.AGF_decomposition import agf_decomposition

        base_visible, detail_visible = agf_decomposition(visible, **AGF_CONFIG)
        base_infrared, detail_infrared = agf_decomposition(infrared, **AGF_CONFIG)
        if detail_fusion == "wle":
            from detail_fusion.MVA_WLE import fuse_detail_dynamic_WLE_im2col_var_optimizer

            detail = fuse_detail_dynamic_WLE_im2col_var_optimizer(
                detail_infrared, detail_visible, **WLE_CONFIG)
        elif detail_fusion == "mle":
            from detail_fusion.MLE import MLE

            detail = MLE(detail_infrared, detail_visible)
        else:
            raise ValueError("detail_fusion must be 'wle' or 'mle'")
    elif mode == "raw":
        base_visible, base_infrared = visible, infrared
        detail = np.zeros_like(visible)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return dict(zip(CHANNELS, (visible, infrared, base_visible, base_infrared, detail)))


class CachedPairs(Dataset):
    def __init__(self, root, split, crop=0, augment=False):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if "detail_fusion" not in self.manifest:
            raise ValueError("Cache does not record detail_fusion; recreate it with data --detail-fusion wle or mle")
        self.records = [item for item in self.manifest["records"] if item["split"] == split]
        if not self.records:
            raise ValueError(f"No {split} images in {root}")
        if crop and crop < 8:
            raise ValueError("crop must be 0 or >= 8")
        self.crop = crop
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with np.load(self.root / record["file"], allow_pickle=False) as stored:
            stack = np.stack([stored[name] for name in CHANNELS]).astype(np.float32)
        if self.crop:
            height, width = stack.shape[-2:]
            if min(height, width) < self.crop:
                raise ValueError(f"{record['id']} is smaller than crop={self.crop}")
            top = int(torch.randint(height - self.crop + 1, ()).item())
            left = int(torch.randint(width - self.crop + 1, ()).item())
            stack = stack[:, top:top + self.crop, left:left + self.crop]
        if self.augment:
            if torch.rand(()) < 0.5:
                stack = stack[:, :, ::-1]
            if torch.rand(()) < 0.5:
                stack = stack[:, ::-1, :]
        tensor = torch.from_numpy(stack.copy())
        result = {name: tensor[position:position + 1] for position, name in enumerate(CHANNELS)}
        result["id"] = record["id"]
        return result


def main():
    parser = argparse.ArgumentParser(description="Cache full-image AGF plus WLE/MLE detail fusion before splitting/cropping")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("agf", "raw"), default="agf")
    parser.add_argument("--detail-fusion", choices=("wle", "mle"), default="wle",
                        help="Detail fusion used with --mode agf; default: wle")
    parser.add_argument("--split", choices=("trainval", "test"), default="trainval")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--val-list", type=Path, help="One image stem per line; keep whole scenes in one split")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--limit", type=int, default=0, help="Smoke checks only; 0 means all pairs")
    args = parser.parse_args()
    pairs = paired_files(args.source)
    if args.limit < 0:
        parser.error("limit must be nonnegative")
    if args.limit:
        pairs = pairs[:args.limit]
    validation = set()
    if args.split == "trainval":
        if len(pairs) < 2 or not 0 < args.val_fraction < 1:
            parser.error("trainval requires >= 2 pairs and val-fraction in (0, 1)")
        names = [name for name, _, _ in pairs]
        if args.val_list:
            validation = set(args.val_list.read_text(encoding="utf-8").splitlines())
            if not validation or not validation < set(names):
                parser.error("val-list must be a nonempty proper subset of image IDs")
        else:
            shuffled = np.random.default_rng(args.seed).permutation(names)
            count = min(len(names) - 1, max(1, round(len(names) * args.val_fraction)))
            validation = set(shuffled[:count].tolist())
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    seen = set()
    for index, (name, visible, infrared) in enumerate(pairs):
        digest = sha256(visible) + sha256(infrared)
        if digest in seen:
            raise ValueError(f"Duplicate image pair: {name}; remove duplicates before splitting")
        seen.add(digest)
        split = "test" if args.split == "test" else ("val" if name in validation else "train")
        relative = f"{index:06d}.npz"
        arrays = prepare_pair(visible, infrared, args.mode, args.detail_fusion)
        np.savez_compressed(args.output / relative, **arrays)
        records.append(dict(id=name, split=split, file=relative, digest=digest))
        print(f"[{index + 1}/{len(pairs)}] {split}: {name}", flush=True)
    manifest = dict(mode=args.mode, detail_fusion=args.detail_fusion if args.mode == "agf" else "none",
                    detail_fusion_config=WLE_CONFIG if args.mode == "agf" and args.detail_fusion == "wle" else None,
                    agf=AGF_CONFIG if args.mode == "agf" else None,
                    source=str(args.source.resolve()), seed=args.seed, records=records)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()