import numpy as np
from Indicator.utils import *
from Indicator.structural_similarity import *

"""
    Calculate sim(A,B,F) = covariance(AF) / covariance(AF) + covariance(BF) and if sim < 0 then sim = 0, if sim > 1 then sim = 1
"""
def calculate_sim(image1, image2, fused):
    # Convert image by divide in 255
    image1 = image1 / 255.0
    image2 = image2 / 255.0
    fused = fused / 255.0

    meanA = np.mean(image1)
    meanB = np.mean(image2)
    meanF = np.mean(fused)
    covAF = np.sum((image1 - meanA) * (fused - meanF)) / (image1.shape[0] * image1.shape[1] - 1)
    covBF = np.sum((image2 - meanB) * (fused - meanF)) / (image1.shape[0] * image1.shape[1] - 1)

    sim = covAF / (covAF + covBF + 1e-10)
    sim = max(0, sim)
    sim = min(1, sim)

    return sim

"""
    Calculate Cvejie index for 2 image input and image fused
"""
def calculate_cvejie_metric(image1, image2, fused, window_size = 8):
    # Convert image to grayscale
    image1 = convert_to_grayscale(image1)
    image2 = convert_to_grayscale(image2)
    fused = convert_to_grayscale(fused)

    # Iterate through all windows
    Qc = 0
    Wc = 0
    for i in range(0, image1.shape[0], window_size):
        for j in range(0, image1.shape[1], window_size):
            # Get window
            window1 = image1[i:i+window_size, j:j+window_size]
            window2 = image2[i:i+window_size, j:j+window_size]
            window_fused = fused[i:i+window_size, j:j+window_size]

            # Calculate sim
            sim = calculate_sim(window1, window2, window_fused)

            # Calculate UIQI
            UIQI1 = calculate_universal_image_quality_index(window1, window_fused)
            UIQI2 = calculate_universal_image_quality_index(window2, window_fused)

            # Calculate Qc
            Qc += sim * UIQI1 + (1 - sim) * UIQI2
            Wc += 1
    
    Qc = Qc / Wc
    return Qc

            