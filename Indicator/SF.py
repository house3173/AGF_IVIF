import numpy as np
from Indicator.utils import *

"""
    Calculate “spatial frequency” to measure the activity level of an image
"""
def calculate_spatial_frequency_old(image, w_d = 1/np.sqrt(2)):
    # Convert image to grayscale then divide by 255
    image = convert_to_grayscale(image)
    image = image / 255.0

    # Calculate RF: sqrt((1/MN) * sum((I(i,j)-I(i,j-1))^2))
    RF = np.sqrt(np.mean((image[:, 1:] - image[:, :-1])**2))

    # Calculate CF: sqrt((1/MN) * sum((I(i,j)-I(i-1,j))^2))
    CF = np.sqrt(np.mean((image[1:, :] - image[:-1, :])**2))

    # Calculate MDF: sqrt(w_d * (1/MN) * sum((I(i,j)-I(i-1,j-1))^2))
    MDF = np.sqrt(w_d * np.mean(np.square(image[1:, 1:] - image[:-1, :-1])))

    # Calculate SDF: sqrt(w_d * (1/MN) * sum((I(i,j)-I(i-1,j+1))^2))
    SDF = np.sqrt(w_d * np.mean(np.square(image[1:, :-1] - image[:-1, 1:])))

    # Calculate SF: sqrt(RF^2 + CF^2 + MDF^2 + SDF^2)
    SF = np.sqrt(RF**2 + CF**2 + MDF**2 + SDF**2)

    return SF

def calculate_spatial_frequency(image):
    """
    Tính toán Spatial Frequency (SF) của một ảnh đầu vào.
    """
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Chuyển ảnh về grayscale nếu chưa

    image = convert_to_grayscale(image)
    # image = image / 255.0

    M, N = image.shape
    
    # Tính toán RF (Row Frequency)
    RF = np.sqrt(np.sum((image[1:, :] - image[:-1, :]) ** 2) / (M * N))
    
    # Tính toán CF (Column Frequency)
    CF = np.sqrt(np.sum((image[:, 1:] - image[:, :-1]) ** 2) / (M * N))
    
    # Tính toán SF
    SF = np.sqrt(RF ** 2 + CF ** 2)
    
    return SF

"""
    Calculate "spatial frequency" between two images (reference)
"""
def calculate_spatial_frequency_ref(image1, image2, w_d = 1/np.sqrt(2)):
    image1 = convert_to_grayscale(image1)
    image1 = image1 / 255.0
    image2 = convert_to_grayscale(image2)
    image2 = image2 / 255.0

    # Calculate Reference spatial frequency between image1 and image2: Gradient_R = max(abs(Grad1), abs(Grad2))
    # Example: RF_Gradient(i,j) = max(abs([I1(i,j)-I1(i,j-1)]), abs([I2(i,j)-I2(i,j-1)])) => RF = sqrt((1/MN) * sum(RF_Gradient(i,j)^2))
    RF_Gradient = np.maximum(
        np.abs(image1[:, 1:] - image1[:, :-1]),
        np.abs(image2[:, 1:] - image2[:, :-1])
    )
    RFR = np.sqrt(np.mean(np.square(RF_Gradient)))

    CF_Gradient = np.maximum(
        np.abs(image1[1:, :] - image1[:-1, :]),
        np.abs(image2[1:, :] - image2[:-1, :])
    )
    CFR = np.sqrt(np.mean(np.square(CF_Gradient)))

    MDF_Gradient = np.maximum(
                        np.abs(image1[1:, 1:] - image1[:-1, :-1]),
                        np.abs(image2[1:, 1:] - image2[:-1, :-1])
                    )
    MDFR = np.sqrt(w_d * np.mean(np.square(MDF_Gradient)))

    SDF_Gradient = np.maximum(
                        np.abs(image1[1:, :-1] - image1[:-1, 1:]),
                        np.abs(image2[1:, :-1] - image2[:-1, 1:])
                    )
    SDFR = np.sqrt(w_d * np.mean(np.square(SDF_Gradient)))

    SFR = np.sqrt(RFR**2 + CFR**2 + MDFR**2 + SDFR**2)

    return SFR

"""
    Calculate "spatial frequency" between two images input and fused image
"""
def calcualte_spatial_frequency_fusion(image1, image2, fused_image, w_d = 1/np.sqrt(2)):
    SFR = calculate_spatial_frequency_ref(image1, image2, w_d)
    SFF = calculate_spatial_frequency(fused_image, w_d)
    result = (SFF - SFR) / SFR

    return result