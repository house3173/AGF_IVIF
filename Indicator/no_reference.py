import numpy as np
import cv2
from Indicator.utils import * 
from scipy.signal import convolve2d

"""
    Calculate Entropy (EN) of an image
"""
def calculate_entropy(image, bins=256):
    histogram = calculate_histogram(image, bins)
    entropy = -np.sum(histogram * np.log2(histogram + 1e-10))  
    return entropy

"""
    Calculate Average Gradient (AG) of an image
"""
def calculate_average_gradient(image, solution = 'none'):
    image = convert_to_grayscale(image)

    if solution == 'none':
        # formula: mean(sqrt(((I(x+1, y) - I(x, y))^2 + (I(x, y+1) - I(x, y))^2)/2))
        # Đồng bộ kích thước của dx và dy
        dx = image[:-1, 1:] - image[:-1, :-1]
        dy = image[1:, :-1] - image[:-1, :-1]
        gradient = np.sqrt((dx ** 2 + dy ** 2) / 2)
        gradient = gradient / 255.0
        average_gradient = np.mean(gradient)
    elif solution == 'sobel':
        dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt((dx ** 2 + dy ** 2) / 2)
        gradient = gradient / 255.0
        average_gradient = np.mean(gradient)

    return average_gradient

"""
    Calculate Standard Deviation (SD) of an image
"""
def calculate_standard_deviation(image):
    image = convert_to_grayscale(image)
    image = image / 255.0

    return np.std(image)

"""
    Calculate Edge Intensity (EI) of an image
"""    
def calculate_edge_intensity(image):
    image = convert_to_grayscale(image)
    image = image / 255.0

    # Sobel filter along x and y axis
    hx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    hy = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]])
    
    EI = np.sqrt(convolve2d(image, hx, mode='valid')**2 + convolve2d(image, hy, mode='valid')**2)

    return np.mean(EI)

"""
    Calculate average light intensity (ALI) of an image
"""
def calculate_average_light_intensity(image):
    image = convert_to_grayscale(image)
    image = image / 255.0

    return np.mean(image)

"""
    Calculate contrast of an image
"""
def calculate_contrast(image):
    image = convert_to_grayscale(image)
    image = image / 255.0

    contrast = np.var(image)

    return contrast