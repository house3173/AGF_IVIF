import numpy as np

def compute_vsm(base, num_bins=256):
    quantized_base = np.round(base * (num_bins - 1)).astype(np.uint8)

    hist = np.bincount(quantized_base.ravel(), minlength=num_bins) / base.size

    V = np.zeros_like(base, dtype=np.float32)
    for j in range(num_bins):
        V += hist[j] * np.abs(base - j / (num_bins - 1)) 

    V = (V - V.min()) / (V.max() - V.min() + 1e-8)
    return V

def low_5(base_ir, base_vi):
    V1 = compute_vsm(base_ir)
    V2 = compute_vsm(base_vi)

    Wb = 0.5 + (V1 - V2) / 2
    Wb = np.clip(Wb, 0, 1)  

    base_fused = Wb * base_ir + (1 - Wb) * base_vi

    return base_fused