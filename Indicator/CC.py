import numpy as np
from Indicator.utils import *

'''
    Calculate  Cross-Correlation (CC) between two images
'''
def calculate_cross_correlation_between_two_images(A, B):
    # Convert images to grayscale
    A = convert_to_grayscale(A)
    B = convert_to_grayscale(B)

    rAB = np.sum((A - np.mean(A)) * (B - np.mean(B))) / np.sqrt(
        np.sum((A - np.mean(A)) ** 2) * np.sum((B - np.mean(B)) ** 2))
    return rAB

'''
    Calculate Cross-Correlation (CC) of two images input and fused image
'''
def calculate_cross_correlation(A, B, F):
    cc_AF = calculate_cross_correlation_between_two_images(A, F)
    cc_BF = calculate_cross_correlation_between_two_images(B, F)
    CC = (cc_AF + cc_BF) / 2

    return CC

'''
    Calculate Sum of Correlation Difference (SCD) of two images input and fused image
'''
def calculate_sum_of_correlation_difference(A, B, F):
    # Convert images to grayscale
    A = convert_to_grayscale(A)
    B = convert_to_grayscale(B)
    F = convert_to_grayscale(F)

    scd_AF = calculate_cross_correlation_between_two_images(F - A, B)
    scd_BF = calculate_cross_correlation_between_two_images(F - B, A)
    SCD = scd_AF + scd_BF

    return SCD



    