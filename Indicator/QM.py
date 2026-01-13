import numpy as np
import pywt
from Indicator.utils import *

"""
    Calculate edge preservation between two images in level s
"""
def calculate_edge_preservation(LHi, HLi, LLi, LHf, HLf, HHf):
    ep_LH = np.exp(-np.abs(LHi - LHf) / 255.0)
    ep_HL = np.exp(-np.abs(HLi - HLf) / 255.0)
    ep_HH = np.exp(-np.abs(LLi - HHf) / 255.0)

    return (ep_LH + ep_HL + ep_HH) / 3

"""
    Calculate multiscale scheme in all levels
"""
def calculate_multiscale_scheme(image1, image2, fused, a_s = [0.5, 0.5], level=2):
    # Convert images to grayscale
    image1 = convert_to_grayscale(image1)
    image2 = convert_to_grayscale(image2)
    fused = convert_to_grayscale(fused)

    # Apply Haar wavelet transform 2-level to 3 images use pywt 
    coeffs1 = pywt.wavedec2(image1, 'haar', level=level)
    coeffs2 = pywt.wavedec2(image2, 'haar', level=level)
    coeffsF = pywt.wavedec2(fused, 'haar', level=level)

    result = 1
    normalized_levels = []
    ep_1s = []
    ep_2s = []

    # Iterate through each level of the wavelet transform and calculate in Hight frequency subbands
    for i in range(1, level+1):
        # Get LH, HL, HH subbands of each image
        LH1, HL1, HH1 = coeffs1[i]
        LH2, HL2, HH2 = coeffs2[i]
        LHf, HLf, HHf = coeffsF[i]

        # Calculate edge preservation in each subband
        ep_1 = calculate_edge_preservation(LH1, HL1, HH1, LHf, HLf, HHf)
        ep_2 = calculate_edge_preservation(LH2, HL2, HH2, LHf, HLf, HHf)

        ep_1s.append(ep_1)
        ep_2s.append(ep_2)

        # Calculate high-frequency energy
        energy_1 = (LH1**2 + HL1**2 + HH1**2) / 255.0 ** 2
        energy_2 = (LH2**2 + HL2**2 + HH2**2) / 255.0 ** 2

        # Calculate normalized performance metric inlevel s:  = sum(ep_1 * energy_1 + ep_2 * energy_2) / sum(energy_1 + energy_2)
        normalized_level = np.sum(ep_1 * energy_1 + ep_2 * energy_2) / np.sum(energy_1 + energy_2)
        normalized_levels.append(normalized_level)

        # Calculate the final result
        result = result * (normalized_level ** a_s[i-1])
    
    return result




