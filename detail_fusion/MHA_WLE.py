import numpy as np

from detail_fusion.MGA_WLE import WLE_dynamic_gauss_im2col
from detail_fusion.MVA_WLE import WLE_dynamic_var_optimized

# Giả định: bạn đã có sẵn 2 hàm:
#   WLE_dynamic_gauss_im2col(matrix: np.ndarray, um: int = 3) -> np.ndarray
#   WLE_dynamic_var_im2col(matrix: np.ndarray, um: int = 3) -> np.ndarray
# Các hàm này phải trả về ma trận kích thước (H, W) (chúng ta đã thiết kế để pad trước).

def WLE_dynamic_hybrid_adaptive(I: np.ndarray, um: int = 3, pad_mode: str = "reflect") -> np.ndarray:
    """
    Hybrid WLE (Gauss + Var) với alpha thích nghi theo gradient magnitude.
    - I: input grayscale image (2D numpy array). Will be cast to float32.
    - um: window size (odd integer, e.g., 3,5,7)
    - pad_mode: passed-through padding mode for gradient handling if needed (unused here but kept for API compatibility)
    Returns:
    - e: hybrid WLE map, same shape as I (H, W)
    """
    if I.ndim != 2:
        raise ValueError("I phải là ảnh 2D grayscale")
    if um % 2 == 0:
        raise ValueError("um phải là số lẻ (odd)")

    # đảm bảo kiểu float32 để tăng tốc và tiết kiệm bộ nhớ
    img = np.asarray(I, dtype=np.float32)
    H, W = img.shape
    r = um // 2

    # (1) Tính E_gauss và E_var bằng hai hàm đã có
    # NOTE: các hàm này giả định đã trả về kích thước (H, W) vì chúng đã pad trước.
    E_gauss = WLE_dynamic_gauss_im2col(img, um=um)  # (H, W)
    E_var   = WLE_dynamic_var_optimized(img, um=um)    # (H, W)

    # (2) Tính gradient magnitude G của ảnh gốc
    # np.gradient trả về các mảng cùng kích thước (H, W)
    Gy, Gx = np.gradient(img)   # chú ý: np.gradient trả về theo thứ tự (axis0, axis1) => (d/dy, d/dx)
    G = np.sqrt(Gx * Gx + Gy * Gy)  # (H, W)

    # (3) Chuẩn hóa alpha về [0,1]
    maxG = G.max()
    if maxG <= 0:
        alpha = np.zeros_like(G, dtype=np.float32)
    else:
        alpha = (G / maxG).astype(np.float32)

    # (4) Kết hợp E_gauss và E_var theo alpha thích nghi
    # đảm bảo cùng dtype
    E_gauss = E_gauss.astype(np.float32)
    E_var   = E_var.astype(np.float32)
    e = alpha * E_gauss + (1.0 - alpha) * E_var

    return e

def fuse_detail_dynamic_WLE_hybrid(detail_ir, detail_vi):
    if detail_ir.ndim != 2 or detail_vi.ndim != 2:
        raise ValueError("Both detail_ir and detail_vi must be 2D arrays")

    # Compute the hybrid WLE for both images
    e_ir = WLE_dynamic_hybrid_adaptive(detail_ir)
    e_vi = WLE_dynamic_hybrid_adaptive(detail_vi)

    # Create a mask where the infrared energy is greater than or equal to the visible energy
    mask_ir = e_ir >= e_vi
    # Fuse the details based on the mask
    fused_detail = np.where(mask_ir, detail_ir.astype(np.float32), detail_vi.astype(np.float32))
    return fused_detail