import numpy as np
from scipy.ndimage import convolve, uniform_filter

def calculate_emls(image):
    h, w = image.shape
    padded = np.pad(image, pad_width=1, mode='reflect')

    # Tính các hướng gradient
    G1 = np.abs(2 * padded[1:-1, 1:-1] - padded[:-2, 1:-1] - padded[2:, 1:-1])        # dọc
    G2 = np.abs(2 * padded[1:-1, 1:-1] - padded[1:-1, :-2] - padded[1:-1, 2:])        # ngang
    G3 = np.abs(2 * padded[1:-1, 1:-1] - padded[:-2, :-2] - padded[2:, 2:])           # chéo /
    G4 = np.abs(2 * padded[1:-1, 1:-1] - padded[:-2, 2:] - padded[2:, :-2])           # chéo \

    # Kết hợp lại theo công thức EMLS
    emls = G1 + G2 + (1 / np.sqrt(2)) * (G3 + G4)

    # Áp dụng bộ lọc Gaussian-like smoothing 3x3
    W = (1 / 16) * np.array([[1, 2, 1],
                             [2, 4, 2],
                             [1, 2, 1]])
    wemls = convolve(emls, W, mode='reflect')
    return wemls

def generate_weight_maps(image1, image2):
    wsem_l1 = calculate_emls(image1)
    wsem_l2 = calculate_emls(image2)
    mapA = (wsem_l1 >= wsem_l2).astype(np.float32)
    mapB = 1.0 - mapA
    return mapA, mapB

def guided_filtering(I, p, r=3, eps=0.01):
    I = I.astype(np.float32)
    p = p.astype(np.float32)

    # Trung bình trong cửa sổ (sử dụng uniform_filter nhanh hơn vòng lặp)
    mean_I = uniform_filter(I, size=r*2+1, mode='reflect')
    mean_p = uniform_filter(p, size=r*2+1, mode='reflect')
    mean_Ip = uniform_filter(I * p, size=r*2+1, mode='reflect')

    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = uniform_filter(I * I, size=r*2+1, mode='reflect')
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = uniform_filter(a, size=r*2+1, mode='reflect')
    mean_b = uniform_filter(b, size=r*2+1, mode='reflect')

    return mean_a * I + mean_b

def high_2(detail_ir, detail_vi):
    mapA, mapB = generate_weight_maps(detail_ir, detail_vi)
    guided_ir = guided_filtering(detail_ir, mapA)
    guided_vi = guided_filtering(detail_vi, mapB)
    detail_fused = guided_ir * detail_ir + guided_vi * detail_vi
    return detail_fused