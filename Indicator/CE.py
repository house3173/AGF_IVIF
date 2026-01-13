import numpy as np
from Indicator.utils import *

"""
    Calculate cross-entropy between two images.
"""
def calculate_cross_entropy_between_two_images(source_image, fused_image, bins=256):
    # Convert images to grayscale
    gray_source = convert_to_grayscale(source_image)
    gray_fused = convert_to_grayscale(fused_image)
    
    # Calculate histograms of the images
    hist_source = calculate_histogram(gray_source, bins)
    hist_fused = calculate_histogram(gray_fused, bins)
    
    # Calculate cross-entropy: cross_entropy = sum(hist_source * log(abs(his_source / hist_fused)))
    cross_entropy = np.sum(hist_source * np.log2(hist_source / (hist_fused + 1e-10) + 1e-10))

    return cross_entropy

"""
    Calculate cross-entropy between two images and the fused image.
"""
def calculate_cross_entropy(source_image1, source_image2, fused_image, bins=256):
    # Calculate cross-entropy between each pair
    CE_A_F = calculate_cross_entropy_between_two_images(source_image1, fused_image, bins)
    CE_B_F = calculate_cross_entropy_between_two_images(source_image2, fused_image, bins)
    
    # Total cross-entropy of two pairs
    CE = (CE_A_F + CE_B_F) / 2
    
    return CE
