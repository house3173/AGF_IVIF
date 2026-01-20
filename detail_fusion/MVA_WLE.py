import numpy as np

# prefer sliding_window_view if available
try:
    from numpy.lib.stride_tricks import sliding_window_view
except Exception:
    sliding_window_view = None
    from numpy.lib.stride_tricks import as_strided

# prefer uniform_filter for box sum (fast native C implementation)
try:
    from scipy.ndimage import uniform_filter
    _HAS_UNIFORM_FILTER = True
except Exception:
    _HAS_UNIFORM_FILTER = False
    from scipy.signal import convolve2d

def _sliding_views(a: np.ndarray, window_shape):
    """
    Return a view of shape (out_h, out_w, win_h, win_w)
    Works with sliding_window_view if available, else as_strided fallback.
    """
    win_h, win_w = window_shape
    H, W = a.shape
    out_h = H - win_h + 1
    out_w = W - win_w + 1
    if sliding_window_view is not None:
        v = sliding_window_view(a, window_shape)  # shape (out_h, out_w, win_h, win_w)
        return v
    else:
        # as_strided fallback (ensure input is contiguous)
        a = np.ascontiguousarray(a)
        s0, s1 = a.strides
        shape = (out_h, out_w, win_h, win_w)
        strides = (s0, s1, s0, s1)
        return as_strided(a, shape=shape, strides=strides)

def WLE_dynamic_var_optimized(matrix: np.ndarray, um: int = 5, pad_mode: str = "reflect") -> np.ndarray:
    """
    Optimized Weighted Laplacian Energy (variance-based) using sliding window views.
    Returns e with same shape as input matrix.
    """
    if matrix.ndim != 2:
        raise ValueError("matrix phải là 2D array")
    if um % 2 == 0:
        raise ValueError("um phải là số lẻ (odd)")

    mat = np.asarray(matrix, dtype=np.float32)
    H, W = mat.shape
    r = um // 2

    # pad so output equals original shape after sliding windows
    mat_pad = np.pad(mat, pad_width=r, mode=pad_mode).astype(np.float32)
    # compute local energy map = sum of squares within window centered at each pixel
    mat_pad_sq = mat_pad * mat_pad

    if _HAS_UNIFORM_FILTER:
        # uniform_filter returns local mean; multiply by window area to get sum
        # note: uniform_filter handles boundaries via mode argument, but mode strings differ.
        # Using the same pad we already applied and mode='constant' on filter is ok if we use the padded array.
        # We will call uniform_filter on the padded array with size=um and mode='constant' (no extra padding).
        # Simpler: use uniform_filter on mat_pad_sq (it returns mean over window), multiply by um*um to get sum.
        # Use mode='nearest' to be safe for padded array, but we've already padded so mode doesn't matter much.
        local_energy_map = uniform_filter(mat_pad_sq, size=um, mode='reflect') * (um * um)
    else:
        # fallback to convolve2d on padded array (slower)
        kernel = np.ones((um, um), dtype=np.float32)
        # convolve2d with mode='same' to keep padded shape
        local_energy_map = convolve2d(mat_pad_sq, kernel, mode='same', boundary='symm').astype(np.float32)

    # take sliding views over the padded arrays
    views = _sliding_views(mat_pad, (um, um))               # shape: (H, W, um, um)
    views_le = _sliding_views(local_energy_map, (um, um))   # shape: (H, W, um, um)

    # compute patch means and mean of squares (per-patch)
    area = float(um * um)
    # sum over last two axes -> shape (H, W)
    sum_view = views.sum(axis=(2, 3), dtype=np.float32)
    sum_view2 = (views * views).sum(axis=(2, 3), dtype=np.float32)

    patch_mean = sum_view / area              # shape (H, W)
    patch_mean2 = sum_view2 / area            # shape (H, W)
    eps = np.finfo(np.float32).eps
    sigma2 = patch_mean2 - patch_mean * patch_mean
    # clamp and add eps for numerical stability, shape (H,W)
    sigma2 = np.maximum(sigma2, 0.0).astype(np.float32) + eps
    denom = 2.0 * sigma2                        # shape (H,W)

    # compute diff: broadcasting patch_mean to window dims
    # shape (H, W, um, um)
    diff = views - patch_mean[..., None, None]

    # exponent = -(diff^2) / denom[...,None,None]
    exponents = - (diff * diff) / denom[..., None, None]

    # compute weights and normalize per patch
    W_mat = np.exp(exponents, dtype=np.float32)  # shape (H, W, um, um)
    sum_W = W_mat.sum(axis=(2, 3), keepdims=True)
    # avoid division by zero
    sum_W[sum_W == 0.0] = eps
    W_norm = W_mat / sum_W                       # normalized per patch

    # compute e per position: sum over um,um of (views_le * W_norm)
    e = (views_le * W_norm).sum(axis=(2, 3), dtype=np.float32)  # shape (H, W)

    return e

def fuse_detail_dynamic_WLE_im2col_var_optimizer(detail_ir, detail_vi):
    if detail_ir.ndim != 2 or detail_vi.ndim != 2:
        raise ValueError("Both detail_ir and detail_vi must be 2D arrays")

    e_ir = WLE_dynamic_var_optimized(detail_ir)
    e_vi = WLE_dynamic_var_optimized(detail_vi)

    mask_ir = e_ir >= e_vi
    fused_detail = np.where(mask_ir, detail_ir.astype(np.float32), detail_vi.astype(np.float32))
    return fused_detail