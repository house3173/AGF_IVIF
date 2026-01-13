import numpy as np
from Indicator.utils import *

"""
    Caculate mutal information between two images
"""
def calculate_mutual_information_between_two_images(image1, image2, bins=256):
    # Transform all images to grayscale
    gray_image1 = convert_to_grayscale(image1)
    gray_image2 = convert_to_grayscale(image2)

    # Calculate MI_A_B based entropy
    entropy_1 = calculate_entropy(gray_image1, bins)
    entropy_2 = calculate_entropy(gray_image2, bins)
    joint_entropy = calculate_joint_entropy(gray_image1, gray_image2, bins)

    MI_A_B = entropy_1 + entropy_2 - joint_entropy

    return MI_A_B

"""
    Caculate mutal information between two images and fused image
"""
def calculate_mutual_information_metric(image1, image2, image_fused, bins=256):
    # Calculate MI_A_F
    MI_A_F = calculate_mutual_information_between_two_images(image1, image_fused, bins)

    # Calculate MI_B_F
    MI_B_F = calculate_mutual_information_between_two_images(image2, image_fused, bins)

    return MI_A_F + MI_B_F

"""
    Caculate normalized mutual information between two images and fused image
"""
def calculate_normalized_mutual_information_metric(image1, image2, image_fused, bins=256):
    # Calculate MI between images
    MI_A_F = calculate_mutual_information_between_two_images(image1, image_fused, bins)
    MI_B_F = calculate_mutual_information_between_two_images(image2, image_fused, bins)

    # Calculate entropy of images
    entropy_1 = calculate_entropy(image1, bins)
    entropy_2 = calculate_entropy(image2, bins)
    entropy_fused = calculate_entropy(image_fused, bins)

    # Calculate NMI
    NMI = 2 * ((MI_A_F / (entropy_1 + entropy_fused + 1e-10)) + (MI_B_F / (entropy_2 + entropy_fused + 1e-10)))

    return NMI
