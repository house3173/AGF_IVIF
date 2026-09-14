import cv2
import numpy as np

def compute_local_energy(image, kernel_size=3):
    """Tính năng lượng cục bộ bằng cách áp dụng tổng lân cận trong cửa sổ kernel_size x kernel_size."""
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    local_energy = cv2.filter2D(image, -1, kernel)
    return local_energy

def compute_structure_tensor(image):
    """Tính tensor cấu trúc bằng cách sử dụng gradient của ảnh."""
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    SS = np.sqrt(grad_x**2 + grad_y**2)  # Độ nổi bật của tensor cấu trúc
    return SS

def compute_LES(image):
    """Tính toán Local Energy Structure (LES) bằng cách kết hợp năng lượng cục bộ và tensor cấu trúc."""
    LE = compute_local_energy(image)
    SS = compute_structure_tensor(image)
    LES = LE * SS  # Nhân từng phần tử để kết hợp thông tin
    return LES

def high_4(detail_ir, detail_vi):
    """Tổng hợp hai thành phần tần số cao từ MRI và PET."""
    LES_MRI = compute_LES(detail_ir)
    LES_PET = compute_LES(detail_vi)
    
    # Tạo bản đồ trọng số M(x, y)
    M = (LES_MRI > LES_PET).astype(np.float32)
    
    # Tổng hợp tần số cao
    detail_fused = M * detail_ir + (1 - M) * detail_vi
    
    return detail_fused