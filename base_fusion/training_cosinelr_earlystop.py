import os
import torch
import matplotlib.pyplot as plt

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from base_fusion.model.convnext_fusion_net_ver0 import ConvNeXtFusionNetVer0
from base_fusion.loss.loss_2 import fusion_loss_vif
from base_fusion.data.data_loader import get_dataloader


# =====================================================
# Train one epoch
# =====================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    loss_sum = {
        "total": 0.0,
        "intensity": 0.0,
        "gradient": 0.0,
        "ssim": 0.0,
    }

    for batch in loader:
        ir = batch["ir"].to(device)
        vi = batch["vi"].to(device)

        fused = model(ir, vi)
        loss, lg, li, ls = criterion(ir, vi, fused)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum["total"] += loss.item()
        loss_sum["gradient"] += lg.item()
        loss_sum["intensity"] += li.item()
        loss_sum["ssim"] += ls.item()

    for k in loss_sum:
        loss_sum[k] /= len(loader)

    return loss_sum


# =====================================================
# Valid one epoch
# =====================================================
@torch.no_grad()
def valid_one_epoch(model, loader, criterion, device):
    model.eval()

    loss_sum = {
        "total": 0.0,
        "intensity": 0.0,
        "gradient": 0.0,
        "ssim": 0.0,
    }

    for batch in loader:
        ir = batch["ir"].to(device)
        vi = batch["vi"].to(device)

        fused = model(ir, vi)
        loss, lg, li, ls = criterion(ir, vi, fused)

        loss_sum["total"] += loss.item()
        loss_sum["gradient"] += lg.item()
        loss_sum["intensity"] += li.item()
        loss_sum["ssim"] += ls.item()

    for k in loss_sum:
        loss_sum[k] /= len(loader)

    return loss_sum


# =====================================================
# Plot losses
# =====================================================
def plot_losses(history, save_dir):
    epochs = range(1, len(history["train"]["total"]) + 1)

    for loss_name in ["total", "intensity", "gradient", "ssim"]:
        plt.figure()
        plt.plot(epochs, history["train"][loss_name], label="Train")
        plt.plot(epochs, history["valid"][loss_name], label="Valid")
        plt.xlabel("Epoch")
        plt.ylabel(f"{loss_name} loss")
        plt.title(loss_name.capitalize())
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"loss_{loss_name}.png"))
        plt.close()


# =====================================================
# Main training function
# =====================================================
def train_fusion(
    root_dir,
    folder_train_outpath,
    epochs=50,
    batch_size=4,
    lr=1e-4,
    device="cuda",
    patience=10,
    min_delta=1e-4
):
    os.makedirs(folder_train_outpath, exist_ok=True)

    # -------------------------
    # Dataloader
    # -------------------------
    train_loader = get_dataloader(root_dir, "train", batch_size)
    valid_loader = get_dataloader(root_dir, "valid", batch_size)

    # -------------------------
    # Model, loss, optimizer
    # -------------------------
    model = ConvNeXtFusionNetVer0().to(device)
    criterion = fusion_loss_vif().to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6
    )

    # -------------------------
    # Loss history
    # -------------------------
    history = {
        "train": {k: [] for k in ["total", "intensity", "gradient", "ssim"]},
        "valid": {k: [] for k in ["total", "intensity", "gradient", "ssim"]},
    }

    best_val = float("inf")
    early_stop_counter = 0

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch [{epoch}/{epochs}]")

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss = valid_one_epoch(
            model, valid_loader, criterion, device
        )

        for k in history["train"]:
            history["train"][k].append(train_loss[k])
            history["valid"][k].append(val_loss[k])

        print(
            f"Train | Total: {train_loss['total']:.4f} | "
            f"I: {train_loss['intensity']:.4f} | "
            f"G: {train_loss['gradient']:.4f} | "
            f"SSIM: {train_loss['ssim']:.4f}"
        )

        print(
            f"Valid | Total: {val_loss['total']:.4f} | "
            f"I: {val_loss['intensity']:.4f} | "
            f"G: {val_loss['gradient']:.4f} | "
            f"SSIM: {val_loss['ssim']:.4f}"
        )

        # -------------------------
        # Scheduler step
        # -------------------------
        scheduler.step()
        print(f"LR: {optimizer.param_groups[0]['lr']:.6e}")

        # -------------------------
        # Early stopping
        # -------------------------
        if val_loss["total"] < best_val - min_delta:
            best_val = val_loss["total"]
            early_stop_counter = 0

            torch.save(
                model.state_dict(),
                os.path.join(folder_train_outpath, "best_model.pth")
            )
            print("✓ Best model saved")

        else:
            early_stop_counter += 1
            print(f"EarlyStopping: {early_stop_counter}/{patience}")

            if early_stop_counter >= patience:
                print("⛔ Early stopping triggered!")
                break

    # -------------------------
    # Save last model & plots
    # -------------------------
    torch.save(
        model.state_dict(),
        os.path.join(folder_train_outpath, "last_model.pth")
    )

    plot_losses(history, folder_train_outpath)


# =====================================================
# Run
# =====================================================
if __name__ == "__main__":
    train_fusion(
        root_dir="C://Users//ADMIN//OneDrive - Hanoi University of Science and Technology//Desktop//AGF_IVIF//data//training_model",
        folder_train_outpath="C://Users//ADMIN//OneDrive - Hanoi University of Science and Technology//Desktop//AGF_IVIF//base_fusion//run//train_1",
        epochs=50,
        batch_size=4,
        lr=1e-4,
        device="cuda",
        patience=10
    )
