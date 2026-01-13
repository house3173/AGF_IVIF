import numpy as np
import cv2
from decomposition.alternatingGuidedFilter import alternating_guided_filter, alternating_guided_filter_numba

def agf_decomposition(
    img,
    sigma_spatial=5,
    sigma_range=0.05,
    iterations=4,
    median_ksize=3,
    numba=True
):
    """AGF decomposition"""
    # Check input image type: if image_path is str, read image
    if isinstance(img, str):
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0

    if not numba:
        base = alternating_guided_filter(
            img,
            sigma_spatial=sigma_spatial,
            sigma_range=sigma_range,
            iterations=iterations,
            median_ksize=median_ksize
        )
    else:
        base = alternating_guided_filter_numba(
            img,
            sigma_spatial=sigma_spatial,
            sigma_range=sigma_range,
            iterations=iterations,
            median_ksize=median_ksize
        )

    detail = img - base

    return base, detail
