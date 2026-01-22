import numpy as np
import torch
import cv2

from base_fusion.model.convnext_fusion_net_ver2 import ConvNeXtFusionNetVer2


def numpy_to_tensor(img_np):
    """
    img_np: (H,W) or (H,W,1)
    return: torch.Tensor (1,1,H,W), float32 [0,1]
    """
    if img_np.ndim == 3:
        img_np = img_np.squeeze(-1)

    img_np = img_np.astype(np.float32)

    if img_np.max() > 1.0:
        img_np /= 255.0

    tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
    return tensor

@torch.no_grad()
def fuse_ir_vi_numpy(
    ir_base: np.ndarray,
    vi_base: np.ndarray,
    model_path: str,
    device: str = "cuda"
) -> np.ndarray:
    """
    IR–VI fusion inference
    """

    # ==========================
    # Load model
    # ==========================
    model = ConvNeXtFusionNetVer2().to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    model.eval()

    # ==========================
    # Preprocess
    # ==========================
    ir = numpy_to_tensor(ir_base).to(device)
    vi = numpy_to_tensor(vi_base).to(device)

    # ==========================
    # Forward
    # ==========================
    fused = model(ir, vi)

    # ==========================
    # Postprocess
    # ==========================
    fused = torch.clamp(fused, 0.0, 1.0)
    fused_np = fused.squeeze().cpu().numpy().astype(np.float32)

    return fused_np

if __name__ == "__main__":
    ir = cv2.imread("ir.png", cv2.IMREAD_GRAYSCALE)
    vi = cv2.imread("vi.png", cv2.IMREAD_GRAYSCALE)

    fused = fuse_ir_vi_numpy(
        ir_base=ir,
        vi_base=vi,
        model_path="path/to/best_model.pth",
        device="cuda"
    )

    cv2.imwrite("fused.png", (fused * 255).astype("uint8"))
