import numpy as np
import torch
import torch.nn as nn

from base_PFCFuse.net import (
    Restormer_Encoder,
    Restormer_Decoder,
    BaseFeatureExtraction,
    DetailFeatureExtraction
)

# ============================================================
# Global cache (load model once)
# ============================================================
_PFCFUSE_MODEL = None


def _load_pfcfuse_model(
    ckpt_path: str = "./base_PFCFuse/models/PFCFuse_epoch_8_01-27-09-12.pth",
    device: str | None = None
):
    """
    Load PFCFuse model exactly as training / test code.
    """
    global _PFCFUSE_MODEL

    if _PFCFUSE_MODEL is not None:
        return _PFCFUSE_MODEL

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------
    # Build network
    # ------------------------------
    Encoder = Restormer_Encoder().to(device)
    Decoder = Restormer_Decoder().to(device)
    BaseFuseLayer = BaseFeatureExtraction(dim=64).to(device)
    DetailFuseLayer = DetailFeatureExtraction(num_layers=1).to(device)

    # ------------------------------
    # Load checkpoint
    # ------------------------------
    ckpt = torch.load(ckpt_path, map_location=device)

    Encoder.load_state_dict(ckpt["DIDF_Encoder"], strict=False)
    Decoder.load_state_dict(ckpt["DIDF_Decoder"], strict=True)
    BaseFuseLayer.load_state_dict(ckpt["BaseFuseLayer"], strict=True)
    DetailFuseLayer.load_state_dict(ckpt["DetailFuseLayer"], strict=True)

    # ------------------------------
    # Eval mode
    # ------------------------------
    Encoder.eval()
    Decoder.eval()
    BaseFuseLayer.eval()
    DetailFuseLayer.eval()

    _PFCFUSE_MODEL = {
        "Encoder": Encoder,
        "Decoder": Decoder,
        "BaseFuseLayer": BaseFuseLayer,
        "DetailFuseLayer": DetailFuseLayer,
        "device": device
    }

    return _PFCFUSE_MODEL


# ============================================================
# Public API
# ============================================================
@torch.no_grad()
def pfcfuse_fuse_gray(
    base_ir: np.ndarray,
    base_vi: np.ndarray
) -> np.ndarray:
    """
    PFCFuse grayscale inference (MSRS-style)

    Args:
        base_ir: (H, W), float32, range [0,1]
        base_vi: (H, W), float32, range [0,1]

    Returns:
        base_fused: (H, W), float32, range [0,1]
    """

    # ------------------------------
    # Sanity check
    # ------------------------------
    assert base_ir.ndim == 2, "base_ir must be (H, W)"
    assert base_vi.ndim == 2, "base_vi must be (H, W)"
    assert base_ir.shape == base_vi.shape, "IR and VIS must have same shape"

    assert base_ir.dtype in [np.float32, np.float64]
    assert base_vi.dtype in [np.float32, np.float64]

    # ------------------------------
    # Load model
    # ------------------------------
    model = _load_pfcfuse_model()
    Encoder = model["Encoder"]
    Decoder = model["Decoder"]
    BaseFuseLayer = model["BaseFuseLayer"]
    DetailFuseLayer = model["DetailFuseLayer"]
    device = model["device"]

    # ------------------------------
    # Prepare input tensor
    # Shape: [1, 1, H, W]
    # ------------------------------
    ir = torch.from_numpy(
        base_ir[np.newaxis, np.newaxis, ...].astype(np.float32)
    ).to(device)

    vi = torch.from_numpy(
        base_vi[np.newaxis, np.newaxis, ...].astype(np.float32)
    ).to(device)

    # ------------------------------
    # Encoder
    # ------------------------------
    feat_v_b, feat_v_d, _ = Encoder(vi)
    feat_i_b, feat_i_d, _ = Encoder(ir)

    # ------------------------------
    # Fusion (EXACT like test script)
    # ------------------------------
    feat_f_b = BaseFuseLayer(feat_v_b + feat_i_b)
    feat_f_d = DetailFuseLayer(feat_v_d + feat_i_d)

    # ------------------------------
    # Decoder
    # ------------------------------
    fused, _ = Decoder(vi, feat_f_b, feat_f_d)

    # ------------------------------
    # Normalize (same as test code)
    # ------------------------------
    # fused = (fused - fused.min()) / (fused.max() - fused.min() + 1e-8)

    # ------------------------------
    # Output
    # ------------------------------
    base_fused = fused.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

    return base_fused
