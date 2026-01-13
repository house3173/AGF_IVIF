import numpy as np
from Indicator.utils import *
from Indicator.structural_similarity import *

"""
    Calculate Yang's metric for 2 image input and image fused use SSIM
"""
def calculate_yang_metric(image1, image2, fused, window_size = 8):
    # Convert image to grayscale
    image1 = convert_to_grayscale(image1)
    image2 = convert_to_grayscale(image2)
    fused = convert_to_grayscale(fused)

    # Iterate through all windows
    Qy = 0
    Wy = 0
    for i in range(0, image1.shape[0], window_size):
        for j in range(0, image1.shape[1], window_size):
            # Get window
            window1 = image1[i:i+window_size, j:j+window_size]
            window2 = image2[i:i+window_size, j:j+window_size]
            window_fused = fused[i:i+window_size, j:j+window_size]

            # Calculate SSIM between window1, window2 and window_fused
            SSIM1 = calculate_structural_similarity_index_measure(window1, window_fused)
            SSIM2 = calculate_structural_similarity_index_measure(window2, window_fused)

            # Calculate SSIM between window1, window2
            SSIM12 = calculate_structural_similarity_index_measure(window1, window2)

            if SSIM12 >= 0.75:
                # Calculate variance of window1, window2
                var1 = np.var(window1)
                var2 = np.var(window2)

                # Calculate lamda(w)
                lamda = var1 / (var1 + var2 + 1e-10)

                Qy += lamda * SSIM1 + (1 - lamda) * SSIM2
            else:
                Qy += np.max([SSIM1, SSIM2])
            
            Wy += 1
    
    Qy = Qy / Wy
    return Qy