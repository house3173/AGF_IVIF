import numpy as np
from scipy.signal import convolve2d

try:
    from numpy.lib.stride_tricks import sliding_window_view
except Exception:
    sliding_window_view = None
    from numpy.lib.stride_tricks import as_strided

def _sliding_patches(a: np.ndarray, window_shape):
    win_h, win_w = window_shape
    if sliding_window_view is not None:
        views = sliding_window_view(a, window_shape)  # (H-win_h+1, W-win_w+1, win_h, win_w)
        Hs, Ws = views.shape[:2]
        patches = views.reshape(Hs * Ws, win_h * win_w).T  # (win_h*win_w, N)
        return patches, Hs, Ws
    else:
        H, W = a.shape
        out_h = H - win_h + 1
        out_w = W - win_w + 1
        s0, s1 = a.strides
        shape = (out_h, out_w, win_h, win_w)
        strides = (s0, s1, s0, s1)
        windows = as_strided(a, shape=shape, strides=strides)
        patches = windows.reshape(out_h * out_w, win_h * win_w).T
        return patches, out_h, out_w

def WLE_dynamic_gauss_im2col(matrix: np.ndarray, um: int = 3) -> np.ndarray:
    """
    Weighted Laplacian Energy (dynamic Gaussian, im2col style).
    matrix: 2D numpy array (H, W)
    um: window size (odd)
    Trả về e có kích thước (H, W), nhờ padding trước khi tính toán.
    """
    if matrix.ndim != 2:
        raise ValueError("matrix phải là 2D array")
    if um % 2 == 0:
        raise ValueError("um phải là số lẻ")

    mat = np.asarray(matrix, dtype=np.float32)
    H, W = mat.shape
    r = um // 2

    # (0) Padding ảnh để giữ kích thước đầu ra = (H,W)
    mat_pad = np.pad(mat, pad_width=r, mode="reflect")  # hoặc "symmetric", "edge" tùy nhu cầu
    H_pad, W_pad = mat_pad.shape

    # (1) local energy
    kernel = np.ones((um, um), dtype=np.float32)
    local_energy = convolve2d(mat_pad * mat_pad, kernel, mode='same', boundary='symm')

    # (2) patches
    patches, out_h, out_w = _sliding_patches(mat_pad, (um, um))
    patches_le, _, _      = _sliding_patches(local_energy, (um, um))

    # (3) sigma cho từng patch
    patch_mean  = patches.mean(axis=0, dtype=np.float32)
    patch_mean2 = (patches * patches).mean(axis=0, dtype=np.float32)
    eps = np.finfo(np.float32).eps
    sigma_vec = np.sqrt(np.maximum(patch_mean2 - patch_mean * patch_mean, 0.0)) + eps

    # (4) dist2
    coords = np.arange(-r, r + 1)
    X, Y = np.meshgrid(coords, coords)
    dist2 = (X * X + Y * Y).reshape(-1).astype(np.float32)

    # (5) Gaussian động
    denom = 2.0 * (sigma_vec * sigma_vec)
    exponents = -(dist2[:, None] / denom[None, :])
    Weights = np.exp(exponents, dtype=np.float32)
    Weights /= Weights.sum(axis=0, keepdims=True)

    # (6) e_vec
    e_vec = np.sum(patches_le * Weights, axis=0, dtype=np.float32)

    # (7) reshape về (out_h, out_w) = (H_pad-2r, W_pad-2r) = (H,W)
    e = e_vec.reshape(out_h, out_w)
    return e

def fuse_detail_dynamic_WLE_im2col(detail_ir, detail_vi):
    if detail_ir.ndim != 2 or detail_vi.ndim != 2:
        raise ValueError("Both detail_ir and detail_vi must be 2D arrays")

    # Compute the dynamic WLE for both images
    e_ir = WLE_dynamic_gauss_im2col(detail_ir)
    e_vi = WLE_dynamic_gauss_im2col(detail_vi)

    # Create a mask where the infrared energy is greater than or equal to the visible energy
    mask_ir = e_ir >= e_vi
    # Fuse the details based on the mask
    fused_detail = np.where(mask_ir, detail_ir, detail_vi)
    return fused_detail