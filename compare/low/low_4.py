import numpy as np

def low_4(base_ir, base_vi, alpha=5):
    # Tính giá trị trung bình (μ) và trung vị (M)
    mu_ir, mu_vis = np.mean(base_ir), np.mean(base_vi)
    M_ir, M_vis = np.median(base_ir), np.median(base_vi)

    # Tính Q
    Q_ir, Q_vis = mu_ir + M_ir, mu_vis + M_vis

    # Tính E theo công thức E = exp(alpha * (B(x,y) - Q))
    E_ir = np.exp(alpha * (base_ir - Q_ir))
    E_vis = np.exp(alpha * (base_vi - Q_vis))

    # Tính ảnh tổng hợp base_fused theo công thức trọng số trung bình
    numerator = E_ir * base_ir + E_vis * base_vi
    denominator = E_ir + E_vis + 1e-8  # Tránh chia cho 0

    base_fused = numerator / denominator

    return base_fused