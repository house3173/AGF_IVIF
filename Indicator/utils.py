import numpy as np
import cv2

"""
   Chuyển đổi ảnh sang dạng grayscale nếu ảnh không phải là ảnh xám.
"""
def convert_to_grayscale(image):
    
    if len(image.shape) == 3: 
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image 
    return gray_image

"""
    Tính histogram của một ảnh grayscale và chuẩn hóa để thu được phân phối xác suất.
"""
def calculate_histogram(image, bins=256):
    
    histogram, _ = np.histogram(image, bins=bins, range=(0, bins-1), density=True)
    return histogram

"""
    Tính histogram chung của hai ảnh grayscale và chuẩn hóa để thu được phân phối xác suất chung.
"""
def calculate_joint_histogram(image1, image2, bins=256):
    
    joint_histogram, _, _ = np.histogram2d(
        image1.ravel(), image2.ravel(), bins=bins, range=[[0, bins-1], [0, bins-1]], density=True
    )
    return joint_histogram

"""
    Tính entropy của một ảnh grayscale.
"""
def calculate_entropy(image, bins=256):
    histogram = calculate_histogram(image, bins)
    entropy = -np.sum(histogram * np.log2(histogram + 1e-10))  
    return entropy

"""
    Tính entropy chung của hai ảnh grayscale.
"""
def calculate_joint_entropy(image1, image2, bins=256):
    joint_histogram = calculate_joint_histogram(image1, image2, bins)
    entropy = -np.sum(joint_histogram * np.log2(joint_histogram + 1e-10))  
    return entropy

"""
    Tính entropy Tsallis cho ảnh grayscale với q cho trước.
"""
def calculate_tsallis_entropy(image, q=1.2):
    histogram = calculate_histogram(image)
    tsallis_entropy = 1 / (q - 1) * (1 - np.sum(histogram ** q))
    return tsallis_entropy

"""
    Tính entropy của một ảnh grayscale với số lượng bins cho trước và logarit cơ số bins.
"""
def calculate_entropy_binned(image, bins=256):
    histogram = calculate_histogram(image, bins)
    entropy = -np.sum(histogram * np.log2(histogram + 1e-10) / np.log2(bins))  
    return entropy

"""
    Tính entropy chung của hai ảnh grayscale với số lượng bins cho trước và logarit cơ số bins.
"""
def calculate_joint_entropy_binned(image1, image2, bins=256):
    joint_histogram = calculate_joint_histogram(image1, image2, bins)
    entropy = -np.sum(joint_histogram * np.log2(joint_histogram + 1e-10) / np.log2(bins))  
    return entropy

""" 
    Edge extraction with Sobel (sb), Canny (cn), and Laplacian of Gaussian (log)
"""
def edge_extraction(image, solution = 'cn'):
    image = convert_to_grayscale(image)

    if solution == 'sb':
        # Using Sobel edge detection
        edge = cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=3)
    elif solution == 'cn':
        # Using Canny edge detection
        edge = cv2.Canny(image, 100, 200)
    elif solution == 'log':
        # Using Laplacian of Gaussian edge detection
        edge = cv2.Laplacian(image, cv2.CV_64F)
    
    return edge
