from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .fusion_model import DEFAULT_CHECKPOINT, load_model


@lru_cache(maxsize=2)
def _cached_model(path, modified_ns, device):
    return load_model(path, device=device)


def clear_model_cache():
    _cached_model.cache_clear()


@torch.inference_mode()
def pfcfuse_fuse_gray(base_ir: np.ndarray, base_vi: np.ndarray, *,
                      ckpt_path=None, device=None, baseline=False) -> np.ndarray:
    """Fuse aligned float [0, 1] arrays; pass a trained checkpoint explicitly."""
    for image in (base_ir, base_vi):
        if not isinstance(image, np.ndarray) or image.ndim != 2:
            raise ValueError("Inputs must be 2D numpy arrays")
        if min(image.shape) < 8 or not np.isfinite(image).all():
            raise ValueError("Inputs must be finite and at least 8 x 8")
        if image.min() < 0 or image.max() > 1:
            raise ValueError("Inputs must be in [0, 1]; divide uint8 images by 255")
    if base_ir.shape != base_vi.shape:
        raise ValueError("IR and VIS shapes must match")
    path = Path(ckpt_path or DEFAULT_CHECKPOINT).resolve(strict=True)
    device = str(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _cached_model(str(path), path.stat().st_mtime_ns, device)
    infrared = torch.from_numpy(np.ascontiguousarray(base_ir, dtype=np.float32))[None, None].to(device)
    visible = torch.from_numpy(np.ascontiguousarray(base_vi, dtype=np.float32))[None, None].to(device)
    fused = model(visible, infrared, baseline=baseline)
    return fused[0, 0].cpu().numpy().astype(np.float32)
