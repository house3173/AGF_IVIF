import numpy as np
from Indicator.utils import *

"""
    Cacular NCC between two images.
"""
def calculate_nonlinear_correlation_coefficient(image1, image2):
    # Convert images to grayscale
    image1_gray = convert_to_grayscale(image1)
    image2_gray = convert_to_grayscale(image2)

    # Calculate entropy of images with binned
    entropy_1 = calculate_entropy_binned(image1_gray)
    entropy_2 = calculate_entropy_binned(image2_gray)
    entropy_joint = calculate_joint_entropy_binned(image1_gray, image2_gray)

    # Calculate NCC
    NCC = entropy_1 + entropy_2 - entropy_joint

    return NCC

"""
    Calculate Nonlinear Correlation Information Entropy (NCIE) between two images and fused image.
"""
def calculate_nonlinear_correlation_information_entropy(image1, image2, image_fused):
    # Build nonlinear correlation coefficient matrix
    R = np.array([
        [1, calculate_nonlinear_correlation_coefficient(image1, image2), calculate_nonlinear_correlation_coefficient(image1, image_fused)],
        [calculate_nonlinear_correlation_coefficient(image2, image1), 1, calculate_nonlinear_correlation_coefficient(image2, image_fused)],
        [calculate_nonlinear_correlation_coefficient(image_fused, image1), calculate_nonlinear_correlation_coefficient(image_fused, image2), 1]
    ])

    # Calculate eigenvalues of matrix R
    eigenvalues = np.linalg.eigvals(R)

    # Calculate NCIE
    NCIE = 1 + sum(
        (eigenvalue / 3) * (np.log(eigenvalue / 3 + 1e-10) / np.log(256)) for eigenvalue in eigenvalues
    )

    return NCIE
