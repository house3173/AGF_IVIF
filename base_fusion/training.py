import os
import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from base_fusion.model.convnext_fusion_net_ver2 import ConvNeXtFusionNetVer2
from base_fusion.loss.loss import fusion_loss_vif
from base_fusion.data.data_loader import get_dataloader

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        ir = batch["ir"].to(device)
        vi = batch["vi"].to(device)

        fused = model(ir, vi)
        loss, lg, li, ls = criterion(ir, vi, fused)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def valid_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    for batch in tqdm(loader, desc="Valid", leave=False):
        ir = batch["ir"].to(device)
        vi = batch["vi"].to(device)

        fused = model(ir, vi)
        loss, _, _, _ = criterion(ir, vi, fused)

        total_loss += loss.item()

    return total_loss / len(loader)

def train_fusion(
    root_dir,
    folder_train_outpath,
    epochs=50,
    batch_size=4,
    lr=1e-4,
    device="cuda"
):
    os.makedirs(folder_train_outpath, exist_ok=True)

    # =====================
    # Dataloader
    # =====================
    train_loader = get_dataloader(root_dir, "train", batch_size)
    valid_loader = get_dataloader(root_dir, "valid", batch_size)

    # =====================
    # Model & Loss
    # =====================
    model = ConvNeXtFusionNetVer2().to(device)
    criterion = fusion_loss_vif().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # =====================
    # Logs
    # =====================
    train_losses = []
    valid_losses = []

    best_val = float("inf")

    # =====================
    # Training
    # =====================
    for epoch in range(1, epochs + 1):
        print(f"//nEpoch [{epoch}/{epochs}]")

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss = valid_one_epoch(
            model, valid_loader, criterion, device
        )

        train_losses.append(train_loss)
        valid_losses.append(val_loss)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                model.state_dict(),
                os.path.join(folder_train_outpath, "best_model.pth")
            )

    # =====================
    # Save final model
    # =====================
    torch.save(
        model.state_dict(),
        os.path.join(folder_train_outpath, "last_model.pth")
    )

    # =====================
    # Plot loss
    # =====================
    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(valid_losses, label="Valid")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(os.path.join(folder_train_outpath, "loss_curve.png"))
    plt.close()

if __name__ == "__main__":
    train_fusion(
        root_dir="C://Users//ADMIN//OneDrive - Hanoi University of Science and Technology//Desktop//AGF_IVIF//data//training_model",
        folder_train_outpath="C://Users//ADMIN//OneDrive - Hanoi University of Science and Technology//Desktop//AGF_IVIF//base_fusion//run//train_1",
        epochs=50,
        batch_size=4,
        lr=1e-4,
        device="cuda"
    )
