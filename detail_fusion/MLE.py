import numpy as np
from scipy.signal import convolve2d

def get_energy(detail):
    W = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
    return convolve2d(detail**2, W, mode='same', boundary='symm')

def MLE(detail_ir, detail_vi):
    # Tính năng lượng toàn ảnh bằng tích chập
    energy_ir = get_energy(detail_ir)
    energy_vi = get_energy(detail_vi)

    # Tạo ảnh kết quả bằng cách chọn pixel có năng lượng cao hơn
    mask = energy_ir >= energy_vi
    detail_fused = np.where(mask, detail_ir, detail_vi)

    return detail_fused
