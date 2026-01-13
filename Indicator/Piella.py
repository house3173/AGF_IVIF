import numpy as np
from Indicator.utils import *
from Indicator.structural_similarity import *

"""
    Calculate Qs = 1/W * sum(lamda(w)*UIQI(A,F|w) + (1-lamda(w))*UIQI(B,F|w)) with w is all windows 8x8 of image and W is number of windows 
        lamda(w) = s(A|w) / (s(A|w) + s(B|w))  with s(A|w) is variance of A in window w
"""
def calculate_Qs(image1, image2, fused, window_size = 8):
    # Convert image to grayscale
    image1 = convert_to_grayscale(image1)
    image2 = convert_to_grayscale(image2)
    fused = convert_to_grayscale(fused)

    # Iterate through all windows
    Qs = 0
    W = 0
    for i in range(0, image1.shape[0], window_size):
        for j in range(0, image1.shape[1], window_size):
            # Get window
            window1 = image1[i:i+window_size, j:j+window_size]
            window2 = image2[i:i+window_size, j:j+window_size]
            window_fused = fused[i:i+window_size, j:j+window_size]

            # Calculate variance of window1, window2
            var1 = np.var(window1)
            var2 = np.var(window2)

            # Calculate lamda(w)
            lamda = var1 / (var1 + var2 + 1e-10)

            # Calculate UIQI
            UIQI1 = calculate_universal_image_quality_index(window1, window_fused)
            UIQI2 = calculate_universal_image_quality_index(window2, window_fused)

            # Calculate Qs
            Qs += lamda * UIQI1 + (1 - lamda) * UIQI2
            W += 1

    return Qs / W

"""
    Calculate Qw = sum(c(w)* [lamda(w)*UIQI(A,F|w) + (1-lamda(w))*UIQI(B,F|w)] ) with w is all windows 8x8 of image
        lamda(w) = s(A|w) / (s(A|w) + s(B|w))  with s(A|w) is variance of A in window w
        c(w) = max(s(A|w), s(B|w)) / sum(max[s(A|w'), s(B|w')] for all w') 
"""
def calculate_Qw(image1, image2, fused, window_size = 8):
    # Convert image to grayscale
    image1 = convert_to_grayscale(image1)
    image2 = convert_to_grayscale(image2)
    fused = convert_to_grayscale(fused)

    # Iterate through all windows to calculate s(A|w) and s(B|w)
    s_A_w = []
    s_B_w = []
    lamda_w = []
    sum_max_s = 0
    
    for i in range(0, image1.shape[0], window_size):
        for j in range(0, image1.shape[1], window_size):
            # Get window
            window1 = image1[i:i+window_size, j:j+window_size]
            window2 = image2[i:i+window_size, j:j+window_size]

            # Calculate variance of window1, window2
            var1 = np.var(window1)
            var2 = np.var(window2)

            s_A_w.append(var1)
            s_B_w.append(var2)
            lamda_w.append(var1 / (var1 + var2 + 1e-10))

            sum_max_s += max(var1, var2)
    
    Q_w = 0
    index = 0
    for i in range(0, image1.shape[0], window_size):
        for j in range(0, image1.shape[1], window_size):
             # Get window
            window1 = image1[i:i+window_size, j:j+window_size]
            window2 = image2[i:i+window_size, j:j+window_size]
            window_fused = fused[i:i+window_size, j:j+window_size]

            # Calculate UIQI
            UIQI1 = calculate_universal_image_quality_index(window1, window_fused)
            UIQI2 = calculate_universal_image_quality_index(window2, window_fused)

            # Calculate Qw
            c_w = max(s_A_w[index], s_B_w[index]) / sum_max_s
            Q_w += c_w * (lamda_w[index] * UIQI1 + (1 - lamda_w[index]) * UIQI2)
            index += 1
    
    return Q_w

"""
    Calculate Qe = Qw(A,B,F) * Qw(A', B', F') with A', B', F' are edge images of A, B, F
"""
def calculate_Qe(image1, image2, fused, window_size = 8, solution = 'cn'):
    # Convert image to grayscale
    image1 = convert_to_grayscale(image1)
    image2 = convert_to_grayscale(image2)
    fused = convert_to_grayscale(fused)

    # Calculate Qw
    Qw = calculate_Qw(image1, image2, fused, window_size)

    # Calculate edge images
    edge1 = edge_extraction(image1, solution)
    edge2 = edge_extraction(image2, solution)
    edge_fused = edge_extraction(fused, solution)

    # Calculate Qw for edge images
    Qw_edge = calculate_Qw(edge1, edge2, edge_fused, window_size)

    return Qw * Qw_edge





