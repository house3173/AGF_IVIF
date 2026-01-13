import numpy as np
from Indicator.utils import *
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift

"""
    Apply Mannos-Skarison filter to an image.
"""
def mannos_skarison_filter(image):
    # Chuyển đổi ảnh sang miền tần số
    f_transform = fft2(image)
    f_shift = fftshift(f_transform)
    
    # Lấy kích thước ảnh
    rows, cols = image.shape
    cx, cy = cols // 2, rows // 2  # Tâm ảnh
    
    # Tạo bộ lọc Mannos-Skarison
    x = np.linspace(-cx, cx, cols)
    y = np.linspace(-cy, cy, rows)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2) + 1e-5  # Tránh chia cho 0
    filter_ms = (2.6 * (0.0192 + 0.114 * R) * np.exp(-((0.0192 + 0.114 * R) ** 1.1)))
    
    # Áp dụng bộ lọc trong miền tần số
    filtered = f_shift * filter_ms
    
    # Chuyển về miền không gian
    f_ishift = ifftshift(filtered)
    image_filtered = np.abs(ifft2(f_ishift))
    
    return filtered, image_filtered 

"""
    Calculate Similarity between two images using CSF (Contrast Sensitivity Function)
"""
def calculate_similarity_csf_test(image1, image2, csf='mannos-skarison'):
    # Divide image1, image2 by 255
    image1 = image1 / 255.0
    image2 = image2 / 255.0

    # Calculate difference of a linear or nonlinear transformation of the two images
    diff = image1 - image2

    # Convert difference to frequency domain using Fourier Transform
    diff = np.fft.fft2(diff)
    diff = np.fft.fftshift(diff)
    freq_diff = np.abs(diff)

    # Calculate radius of the frequency domain
    h, w = freq_diff.shape
    u = np.fft.fftfreq(h).reshape(-1, 1) * h
    v = np.fft.fftfreq(w).reshape(1, -1) * w
    r = np.sqrt(u**2 + v**2)

    # Calculate CSF
    if csf == 'mannos-skarison':
        # Calculate CSF using Mannos-Skarison model
        csf = 2.6 * (0.0192 + 0.114 * r) * np.exp(- (0.114 * r)**1.1)
    elif csf == 'daly':
        # Calculate CSF using Daly model
        csf = (0.008 / (r ** 3) + 1) ** (-0.2) * 1.42 * r * np.exp(-0.3 * r * np.sqrt(1 + 0.06 * np.exp(0.3 * r)))
    elif csf == 'ahumada':
        # Calculate CSF using Ahumada model
        a_c = 1
        a_s = 0.685
        f_c = 97.3227
        f_s = 12.1653
        csf = a_c * np.exp((r / f_c)**2) - a_s * np.exp((r / f_s)**2)
    else:
        raise ValueError('Invalid CSF model')
    
    # Calculate weighted frequency domain
    weighted_freq_diff = freq_diff * csf

    # Calculate similarity
    similarity = np.mean(weighted_freq_diff ** 2)

    return similarity 
         
def calculate_similarity_csf(image1, image2, csf='mannos-skarison'):
    # Calculate difference of a linear or nonlinear transformation of the two images
    diff = np.abs(image1 - image2)

    # Apply Mannos-Skarison filter to difference image
    filter, diff_filtered = mannos_skarison_filter(diff)

    filter = np.abs(filter / 255.0)
    diff_filtered = diff_filtered / 255.0

    # Calculate similarity
    similarity = np.mean(filter ** 2)

    return similarity

"""
    Calculate edge strength of an image using Sobel edge detection. 
"""
def edge_strength(image):
    # Convert image to grayscale
    image = convert_to_grayscale(image)

    # Apply Sobel edge detection
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    edge = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    return edge

"""
    Calculata Chen-Varshney index for 2 image input and image fused
"""
def calculate_chen_varshney(image1, image2, fused, alpha = 2, window_size = 16, edge_solution = 'sb', csf='mannos-skarison'):
    # Convert image to grayscale
    image1 = convert_to_grayscale(image1)
    image2 = convert_to_grayscale(image2)
    fused = convert_to_grayscale(fused)

    # # Extract edge image
    # edge1 = edge_extraction(image1, edge_solution)
    # edge1 = edge1 / 255.0
    # edge2 = edge_extraction(image2, edge_solution)
    # edge2 = edge2 / 255.0

    # Get edge strength map
    edge1 = edge_strength(image1)
    edge2 = edge_strength(image2)

    # Iterate through all windows
    Q_cv_numerator = 0
    Q_cv_denominator = 0

    for i in range(0, image1.shape[0], window_size):
        for j in range(0, image1.shape[1], window_size):
             # Get window
            window1 = image1[i:i+window_size, j:j+window_size]
            window2 = image2[i:i+window_size, j:j+window_size]
            window_fused = fused[i:i+window_size, j:j+window_size]

            # Get edge window
            edge_window1 = edge1[i:i+window_size, j:j+window_size]
            edge_window2 = edge2[i:i+window_size, j:j+window_size]

            # Calculate local region saliency
            saliency1 = np.sum(edge_window1 ** alpha)
            saliency2 = np.sum(edge_window2 ** alpha)

            # Calculate similarity between window1, window2 and window_fused
            similarity1 = calculate_similarity_csf(window1, window_fused, csf)
            similarity2 = calculate_similarity_csf(window2, window_fused, csf)

            # Calculate Q_cv_numerator, Q_cv_denominator
            Q_cv_numerator += (saliency1 * similarity1 + saliency2 * similarity2)
            Q_cv_denominator += (saliency1 + saliency2)

    # Calculate Q_cv
    Q_cv = Q_cv_numerator / Q_cv_denominator

    return Q_cv



