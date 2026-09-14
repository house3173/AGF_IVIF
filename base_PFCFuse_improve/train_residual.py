import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import CachedPairs, sha256
from .evaluation import eligible, evaluate, to_device
from .fusion_model import DEFAULT_CHECKPOINT, FORMAT, load_model
from .objective import FusionObjective


def save_checkpoint(path, model, metadata, epoch, means, optimizer=None):
    payload = dict(format=FORMAT, limit=model.limit, model=model.state_dict(),
                   metadata=metadata, epoch=epoch, validation=means)
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    parser = argparse.ArgumentParser(description="Train bounded residual fusion with frozen original PFCFuse")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--limit", type=float, default=0.25)
    parser.add_argument("--adapt", choices=("both", "base", "detail"), default="both")
    parser.add_argument("--finetune-fusion-lr", type=float, default=0.0,
                        help="If > 0, also fine-tune the pretrained BaseFuseLayer/DetailFuseLayer "
                             "at this (lower) learning rate, gated by the same validation criteria")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-train-batches", type=int, default=0, help="Smoke checks only; 0 means all")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.crop < 8 or args.lr <= 0 or args.max_train_batches < 0:
        parser.error("Invalid epoch, batch, crop, learning rate or batch limit")
    if args.finetune_fusion_lr < 0:
        parser.error("finetune-fusion-lr must be nonnegative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    train_set = CachedPairs(args.cache, "train", crop=args.crop, augment=True)
    val_set = CachedPairs(args.cache, "val")
    train_digests = {item["digest"] for item in train_set.records}
    val_digests = {item["digest"] for item in val_set.records}
    if train_digests & val_digests:
        parser.error("Train/validation image pairs overlap")
    payload = torch.load(args.pretrained, map_location="cpu", weights_only=True)
    if "format" in payload:
        parser.error("Start from the ORIGINAL pretrained checkpoint, not a previously selected candidate")
    model = load_model(args.pretrained, device=args.device, limit=args.limit)
    for name, adapter in model.adapters.items():
        adapter.requires_grad_(args.adapt in ("both", name))
    fusion_layers = [model.backbone["BaseFuseLayer"], model.backbone["DetailFuseLayer"]]
    if args.finetune_fusion_lr > 0:
        for layer in fusion_layers:
            layer.requires_grad_(True)
    adapter_parameters = [parameter for parameter in model.adapters.parameters() if parameter.requires_grad]
    fusion_parameters = ([parameter for layer in fusion_layers for parameter in layer.parameters()]
                         if args.finetune_fusion_lr > 0 else [])
    parameters = adapter_parameters + fusion_parameters
    if not parameters:
        parser.error("No trainable parameters selected; check --adapt")
    param_groups = [dict(params=adapter_parameters, lr=args.lr)]
    if fusion_parameters:
        param_groups.append(dict(params=fusion_parameters, lr=args.finetune_fusion_lr))
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr / 20)
    objective = FusionObjective().to(args.device)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, generator=generator)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = dict(mode=train_set.manifest["mode"], detail_fusion=train_set.manifest["detail_fusion"],
                    detail_fusion_config=train_set.manifest["detail_fusion_config"],
                    agf=train_set.manifest["agf"],
                    pretrained_sha256=sha256(args.pretrained),
                    manifest_sha256=sha256(args.cache / "manifest.json"),
                    used_digests=sorted(train_digests | val_digests),
                    arguments={name: str(value) if isinstance(value, Path) else value
                               for name, value in vars(args).items()},
                    trainable_parameters=sum(parameter.numel() for parameter in parameters))
    initial, _ = evaluate(model, val_loader, args.device)
    save_checkpoint(args.output / "selected.pth", model, metadata, -1, initial)
    best_qabf = initial["original_Qabf"]
    selected_epoch = -1
    history = []
    print(f"Baseline validation: {json.dumps(initial)}", flush=True)
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for step, batch in enumerate(train_loader):
            batch = to_device(batch, args.device)
            features = model.encode(batch["base_visible"], batch["base_infrared"])
            with torch.no_grad():
                teacher = model.decode(batch["base_visible"], features, baseline=True)
            optimizer.zero_grad(set_to_none=True)
            fused = model.decode(batch["base_visible"], features)
            loss = objective(fused, batch, teacher)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch={epoch}, step={step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            if args.max_train_batches and step + 1 >= args.max_train_batches:
                break
        scheduler.step()
        means, _ = evaluate(model, val_loader, args.device)
        accepted = eligible(means) and means["improved_Qabf"] > best_qabf
        if accepted:
            best_qabf = means["improved_Qabf"]
            selected_epoch = epoch
            save_checkpoint(args.output / "selected.pth", model, metadata, epoch, means)
        save_checkpoint(args.output / "last.pth", model, metadata, epoch, means, optimizer)
        record = dict(epoch=epoch, training_loss=float(np.mean(losses)), selected=accepted, validation=means)
        history.append(record)
        summary = dict(metadata=metadata, baseline=initial, selected_epoch=selected_epoch,
                       selection="baseline_retained" if selected_epoch < 0 else "validation_candidate",
                       history=history)
        (args.output / "history.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(record), flush=True)
    print(f"Selected epoch: {selected_epoch} (-1 means original retained). Test before deployment.")


if __name__ == "__main__":
    main()