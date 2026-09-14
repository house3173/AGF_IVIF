import numpy as np

def low_2(base_ir, base_vi):
    base_fused = np.maximum(base_ir, base_vi)
    return base_fused