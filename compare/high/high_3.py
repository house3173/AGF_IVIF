import numpy as np
from scipy.ndimage import gaussian_filter

def pcnn_transform(image, alpha_F=0.5, alpha_L=0.5, alpha_Theta=0.2, beta=0.2, V=1, iterations=100):

    # image = image.astype(np.float32) / 255.0  # Chuẩn hóa về miền [0,1]
    height, width = image.shape
    F = np.zeros((height, width))
    L = np.zeros((height, width))
    Theta = np.ones((height, width))
    Y = np.zeros((height, width))

    for _ in range(iterations):
        F = np.exp(-alpha_F) * F + image + V * gaussian_filter(Y, sigma=1)
        L = np.exp(-alpha_L) * L + V * gaussian_filter(Y, sigma=1)
        U = F * (1 + beta * L)
        
        fired = U > Theta
        Y[fired] = 1
        Theta = np.exp(-alpha_Theta) * Theta + fired

    return Y

def high_3(detail_ir, detail_vi, epsilon=0.1):
    # Tính toán bản đồ kích hoạt từ PCNN
    T_A = pcnn_transform(detail_ir)
    T_B = pcnn_transform(detail_vi)

    # Tạo ảnh tổng hợp
    detail_fused = np.zeros_like(detail_ir)

    # Trường hợp chênh lệch lớn, chọn giá trị lớn hơn
    mask_A = T_A > T_B
    mask_B = T_A < T_B
    detail_fused[mask_A] = detail_ir[mask_A]
    detail_fused[mask_B] = detail_vi[mask_B]

    # Trường hợp chênh lệch nhỏ, áp dụng trung bình có trọng số
    diff_mask = np.abs(T_A - T_B) <= epsilon
    MSMG_A = np.abs(detail_ir)  # Độ mạnh biên của ảnh A
    MSMG_B = np.abs(detail_vi)  # Độ mạnh biên của ảnh B

    lambda_1 = MSMG_A / (MSMG_A + MSMG_B + 1e-6)
    lambda_2 = MSMG_B / (MSMG_A + MSMG_B + 1e-6)

    detail_fused[diff_mask] = lambda_1[diff_mask] * detail_ir[diff_mask] + lambda_2[diff_mask] * detail_vi[diff_mask]

    return detail_fused