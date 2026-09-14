import numpy as np

def high_1(detail_ir, detail_vi):
    detail_fused = np.maximum(detail_ir, detail_vi)
    return detail_fused