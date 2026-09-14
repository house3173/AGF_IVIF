import numpy as np
import math
from scipy.signal import convolve2d

def get_energy_distance_kernel(base):
    W = np.array([[1/(1+math.sqrt(2)), 1/2, 1/(1+math.sqrt(2))], [1/2, 1, 1/2], [1/(1+math.sqrt(2)), 1/2, 1/(1+math.sqrt(2))]])
    return convolve2d(base**2, W, mode='same', boundary='symm')

def low_6(base_ir, base_vi):
    # Tính năng lượng toàn ảnh bằng tích chập
    energy_ir = get_energy_distance_kernel(base_ir)
    energy_vi = get_energy_distance_kernel(base_vi)

    # Tạo ảnh kết quả bằng cách chọn pixel có năng lượng cao hơn
    mask = energy_ir >= energy_vi
    base_fused = np.where(mask, base_ir, base_vi)

    return base_fused
