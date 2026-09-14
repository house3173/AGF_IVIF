import numpy as np
import cv2
from decomposition.jointBilateralFilter import joint_bilateral_filter, joint_bilateral_filter_numba

def alternating_guided_filter(
    I,
    sigma_spatial=5.0,
    sigma_range=0.05,
    iterations=4,
    median_ksize=3
):
    """
    Alternating Guided Filter (AGF) – paper-original version
    I: input image, float32, [0,1]
    """
    I = I.astype(np.float32)

    # Step 1: Initialize G0 as constant image
    G = np.full_like(I, np.mean(I), dtype=np.float32)

    for _ in range(iterations):
        # Step 3: JBF (input=I, guidance=G)
        G = joint_bilateral_filter(
            p=I,
            I=G,
            sigma_spatial=sigma_spatial,
            sigma_range=sigma_range
        )

        # Step 4: JBF (input=G, guidance=I)
        G = joint_bilateral_filter(
            p=G,
            I=I,
            sigma_spatial=sigma_spatial,
            sigma_range=sigma_range
        )

        # Step 5: Median filter
        G = cv2.medianBlur(
            (G * 255).astype(np.uint8),
            median_ksize
        ).astype(np.float32) / 255.0

    return np.clip(G, 0, 1)

def alternating_guided_filter_numba(
    I,
    sigma_spatial=5,
    sigma_range=0.05,
    iterations=4,
    median_ksize=3
):
    I = I.astype(np.float32)
    G = np.full_like(I, np.mean(I), dtype=np.float32)

    for _ in range(iterations):
        G = joint_bilateral_filter_numba(I, G, sigma_spatial, sigma_range)
        G = joint_bilateral_filter_numba(G, I, sigma_spatial, sigma_range)

        G = cv2.medianBlur(
            (G * 255).astype(np.uint8),
            median_ksize
        ).astype(np.float32) / 255.0

    return np.clip(G, 0, 1)