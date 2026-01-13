import numpy as np
from Indicator.utils import *

"""
    Calculate edge strength and edge orientation of an image using Sobel operator.
"""
def sobel_operator(image):
    # Convert image to grayscale 
    image = convert_to_grayscale(image)

    # Calculate Sobel operator in x and y direction
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate edge strength and edge orientation
    edge_strength = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    edge_orientation = np.arctan2(sobel_y, sobel_x)

    return edge_strength, edge_orientation

"""
    Calculate relative strength and orientation values between two images.
"""
def calculate_relative_strength_orientation(image1, image2):
    # Calculate edge strength and edge orientation of two images
    edge_strength1, edge_orientation1 = sobel_operator(image1)
    edge_strength2, edge_orientation2 = sobel_operator(image2)

    # Calculate relative strength: edge_strength1 / edge_strength2 if edge_strength2 > edge_strength1, otherwise edge_strength2 / edge_strength1
    relative_strength = np.where(edge_strength2 > edge_strength1, edge_strength1 / (edge_strength2 + 1e-10), edge_strength2 / (edge_strength1 + 1e-10))

    # Calculate relative orientation: 1 - (|edge_orientation1 - edge_orientation2| / pi/2)
    relative_orientation = 1 - (np.abs(edge_orientation1 - edge_orientation2) / (np.pi / 2))    

    return relative_strength, relative_orientation

"""
    Calculate Edge information preservation value between two images.
"""
def calculate_edge_information_preservation(image1, image2, gamma_g=0.9994, k_g=15 , sigma_g=0.5, gamma_alpha=0.9879, k_alpha=22, sigma_alpha=0.8):
    # Calculate relative strength and orientation values between two images
    relative_strength, relative_orientation = calculate_relative_strength_orientation(image1, image2)

    # Calculate edge strength preservation value: Q_edge = gamma_g / (1 + exp(k_g * (relative_strength - sigma_g)))
    edge_strength_preservation = gamma_g / (1 + np.exp(k_g * (relative_strength - sigma_g)))

    # Calculate edge orientation preservation value: Q_orientation = gamma_alpha / (1 + exp(k_alpha * (relative_orientation - sigma_alpha)))
    edge_orientation_preservation = gamma_alpha / (1 + np.exp(k_alpha * (relative_orientation - sigma_alpha)))

    # Calculate edge information preservation value: Q = Q_edge * Q_orientation
    edge_information_preservation = edge_strength_preservation * edge_orientation_preservation

    return edge_information_preservation

"""
    Calculate Edge information preservation value between two images input and fused image.
"""
def calculate_edge_information_preservation_fused(image1, image2, fused_image, gamma_g=0.9994, k_g=-15 , sigma_g=0.5, gamma_alpha=0.9879, k_alpha=-22, sigma_alpha=0.8, L = 1.5):
    # Calculate edge information preservation value between two images and fused image
    Q1 = calculate_edge_information_preservation(image1, fused_image, gamma_g, k_g, sigma_g, gamma_alpha, k_alpha, sigma_alpha)
    Q2 = calculate_edge_information_preservation(image2, fused_image, gamma_g, k_g, sigma_g, gamma_alpha, k_alpha, sigma_alpha)

    # Calculate weight of each image in fused image: weight = edge_strength ^ L
    edge_strength_1, _ = sobel_operator(image1)
    weight1 = edge_strength_1 ** L
    edge_strength_2, _ = sobel_operator(image2)
    weight2 = edge_strength_2 ** L

    # Calculate edge information preservation value between two images input and fused image: Q = sum(Q1 * weight1 + Q2 * weight2) / sum(weight1 + weight2)
    edge_information_preservation_fused = np.sum(Q1 * weight1 + Q2 * weight2) / np.sum(weight1 + weight2)

    return edge_information_preservation_fused

    