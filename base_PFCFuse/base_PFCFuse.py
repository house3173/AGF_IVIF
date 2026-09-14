import numpy as np
import torch
import torch.nn as nn

from base_PFCFuse.net import (
    Restormer_Encoder,
    Restormer_Decoder,
    BaseFeatureExtraction,
    DetailFeatureExtraction
)

_PFCFUSE_MODEL = None


def _load_pfcfuse_model(
    ckpt_path="./base_PFCFuse/PFCFuse_IVF.pth",
    device=None
):
    global _PFCFUSE_MODEL

    if _PFCFUSE_MODEL is not None:
        return _PFCFUSE_MODEL

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
    Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
    BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction(dim=64)).to(device)
    DetailFuseLayer = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)

    Encoder.load_state_dict(ckpt["DIDF_Encoder"], strict=False)
    Decoder.load_state_dict(ckpt["DIDF_Decoder"])
    BaseFuseLayer.load_state_dict(ckpt["BaseFuseLayer"])
    DetailFuseLayer.load_state_dict(ckpt["DetailFuseLayer"])

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

@torch.no_grad()
def pfcfuse_fuse_gray(
    base_ir: np.ndarray,
    base_vi: np.ndarray
) -> np.ndarray:
    """
    Args:
        base_ir: (H, W), grayscale, float32, range [0,1]
        base_vi: (H, W), grayscale, float32, range [0,1]

    Returns:
        base_fused: (H, W), grayscale, float32, range [0,1]
    """

    assert base_ir.ndim == 2, "base_ir must be (H, W)"
    assert base_vi.ndim == 2, "base_vi must be (H, W)"
    assert base_ir.shape == base_vi.shape, "IR and VIS must have same shape"

    model = _load_pfcfuse_model()
    Encoder = model["Encoder"]
    Decoder = model["Decoder"]
    BaseFuseLayer = model["BaseFuseLayer"]
    DetailFuseLayer = model["DetailFuseLayer"]
    device = model["device"]

    # ===============================
    # Prepare input tensor
    # ===============================
    ir = base_ir[np.newaxis, np.newaxis, ...].astype(np.float32)
    vi = base_vi[np.newaxis, np.newaxis, ...].astype(np.float32)

    ir = torch.from_numpy(ir).to(device)
    vi = torch.from_numpy(vi).to(device)

    # ===============================
    # Encoder
    # ===============================
    feat_v_b, feat_v_d, _ = Encoder(vi)
    feat_i_b, feat_i_d, _ = Encoder(ir)

    # ===============================
    # Fusion
    # ===============================
    feat_f_b = BaseFuseLayer(feat_v_b + feat_i_b)
    feat_f_d = DetailFuseLayer(feat_v_d + feat_i_d)

    # ===============================
    # Decoder
    # ===============================
    fused, _ = Decoder(vi, feat_f_b, feat_f_d)

    # ===============================
    # Normalize & output
    # ===============================
    # fused = (fused - fused.min()) / (fused.max() - fused.min() + 1e-8)

    base_fused = fused.squeeze().cpu().numpy().astype(np.float32)

    return base_fused
