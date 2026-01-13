import numpy as np
from Indicator.utils import *

"""
    Calculate Tsallis Entropy between two images
"""
def calculate_tsallis_entropy_between_two_images(image1, image2, q=1.2):
    # Transform all images to grayscale
    gray_image1 = convert_to_grayscale(image1)
    gray_image2 = convert_to_grayscale(image2)

    # Calculate entropy of images
    entropy_1 = calculate_entropy(gray_image1)
    entropy_2 = calculate_entropy(gray_image2)
    entropy_fused = calculate_joint_entropy(gray_image1, gray_image2)

    # Calculate Tsallis Entropy
    ''' I = 1 / (1-q) * (1 - Σ (entropy_fused ^ q / entropy_2 * entropy_1 ^ (q-1))) '''
    tsallis_entropy = 1 / (1 - q) * (1 - np.sum((entropy_fused ** q) / (entropy_2 * entropy_1 ** (q - 1))))

    return tsallis_entropy

"""
    Calculate Tsallis Entropy between two images and fused image
"""
def calculate_tsallis_entropy_metric(image1, image2, image_fused, q=1.2):
    # Calculate Tsallis Entropy between image1 and image_fused
    tsallis_entropy_1 = calculate_tsallis_entropy_between_two_images(image1, image_fused, q)

    # Calculate Tsallis Entropy between image2 and image_fused
    tsallis_entropy_2 = calculate_tsallis_entropy_between_two_images(image2, image_fused, q)

    return tsallis_entropy_1 + tsallis_entropy_2

"""
    Calculate normalized Tsallis Entropy between two images and fused image
"""
def calculate_normalized_tsallis_entropy_metric(image1, image2, image_fused, q=1.2):
    # Calculate Tsallis Entropy between images
    tsallis_entropy_1 = calculate_tsallis_entropy_between_two_images(image1, image_fused, q)
    tsallis_entropy_2 = calculate_tsallis_entropy_between_two_images(image2, image_fused, q)
    tsallis_entropy_input = calculate_tsallis_entropy_between_two_images(image1, image2, q)

    # Calculate entropy Tsallis of image1 and image2
    entropy_1 = calculate_tsallis_entropy(image1, q)
    entropy_2 = calculate_tsallis_entropy(image2, q)    

    # Calculate NTE
    NTE = (tsallis_entropy_1 + tsallis_entropy_2) / (entropy_1 + entropy_2 - tsallis_entropy_input)

    return NTE


