import numpy as np
from Indicator.utils import *

"""
    Calculate structural similarity index measure (SSIM) for images A and B
"""
def calculate_structural_similarity_index_measure(image1, image2, alpha = 1, beta = 1, gamma = 1, C1 = 1e-10, C2=1e-10, C3=1e-10):
    # Convert image to grayscale and then divide in 255
    image1 = convert_to_grayscale(image1)
    image1 = image1 / 255.0
    image2 = convert_to_grayscale(image2)
    image2 = image2 / 255.0

    # Calculte average and std of image1, image2 and covariance between image1 - image2
    meanA = np.mean(image1)
    meanB = np.mean(image2)
    stdA = np.std(image1)
    stdB = np.std(image2)
    covAB = np.sum((image1 - meanA) * (image2 - meanB)) / (image1.shape[0] * image1.shape[1] - 1)

    # Calculate SSIM
    luminance = (2 * meanA * meanB + C1) / (meanA ** 2 + meanB ** 2 + C1)
    contrast = (2 * stdA * stdB + C2) / (stdA ** 2 + stdB ** 2 + C2)
    correlation = (covAB + C3) / (stdA * stdB + C3)

    return (luminance ** alpha) * (contrast ** beta) * (correlation ** gamma)

"""
    Calculate structural similarity index measure (SSIM) for 2 image input and image fused
"""
def calculate_structural_similarity_index_measure_fused(image1, image2, image_fused, alpha = 1, beta = 1, gamma = 1, C1 = 1e-10, C2=1e-10, C3=1e-10):
    ssimAF = calculate_structural_similarity_index_measure(image1, image_fused, alpha, beta, gamma, C1, C2, C3)
    ssimBF = calculate_structural_similarity_index_measure(image2, image_fused, alpha, beta, gamma, C1, C2, C3)

    return (ssimAF + ssimBF) / 2

"""
    Calculate universal image quality index (UIQI)
"""
def calculate_universal_image_quality_index(image1, image2):
    # Convert image to grayscale and then divide in 255
    image1 = convert_to_grayscale(image1)
    image1 = image1 / 255.0
    image2 = convert_to_grayscale(image2)
    image2 = image2 / 255.0

    # Calculate average and std of image1, image2 and covariance between image1 - image2
    meanA = np.mean(image1)
    meanB = np.mean(image2)
    stdA = np.std(image1)
    stdB = np.std(image2)
    covAB = np.sum((image1 - meanA) * (image2 - meanB)) / (image1.shape[0] * image1.shape[1] - 1)

    # Calculate UIQI
    uiqi = (4 * covAB * meanA * meanB) / ((meanA ** 2 + meanB ** 2) * (stdA ** 2 + stdB ** 2) + 1e-10)

    return uiqi

"""
    Calculate universal image quality index (UIQI) for 2 image input and image fused
"""
def calculate_universal_image_quality_index_fused(image1, image2, image_fused):
    uiqiAF = calculate_universal_image_quality_index(image1, image_fused)
    uiqiBF = calculate_universal_image_quality_index(image2, image_fused)

    return (uiqiAF + uiqiBF) / 2