import numpy as np
from Indicator.utils import convert_to_grayscale

"""
    Calculate the mean square error between two images.
"""
def compute_mean_square_error_between_two_image(image1, image2):
    # Convert both images to grayscale then divede by 255
    image1 = convert_to_grayscale(image1)
    image1 = image1 / 255.0
    image2 = convert_to_grayscale(image2)
    image2 = image2 / 255.0

    # Calculate the mean square error
    mse = np.mean((image1 - image2) ** 2)

    return mse

"""
    Calculate the mean square error between two images and the fused image.
"""
def compute_mean_square_error(image1, image2, image_fused):
    mse_AF = compute_mean_square_error_between_two_image(image1, image_fused)
    mse_BF = compute_mean_square_error_between_two_image(image2, image_fused)

    mse = (mse_AF + mse_BF) / 2
    return mse

"""
    Calculate the root mean square error between two images and the fused image.
"""
def compute_root_mean_square_error(image1, image2, image_fused):
    mse_AF = compute_mean_square_error_between_two_image(image1, image_fused)
    mse_BF = compute_mean_square_error_between_two_image(image2, image_fused)

    rmse = (np.sqrt(mse_AF) + np.sqrt(mse_BF)) / 2
    return rmse

"""
    Calculate the peak signal-to-noise ratio between two images and the fused image.
"""
def compute_peak_signal_to_noise_ratio(image1, image2, image_fused):
    # Convert image_fused to grayscale
    image_fused = convert_to_grayscale(image_fused)
    image_fused = image_fused / 255.0

    max_pixel = np.max(image_fused)

    # Calculate psnr between image1 and image_fused
    mse1 = compute_mean_square_error_between_two_image(image1, image_fused)
    if mse1 == 0:
        return float('inf')
    psnr1 = 10 * np.log10((max_pixel ** 2) / mse1)

    # Calculate psnr between image2 and image_fused
    mse2 = compute_mean_square_error_between_two_image(image2, image_fused)
    if mse2 == 0:
        return float('inf')
    psnr2 = 10 * np.log10((max_pixel ** 2) / mse2)

    psnr = (psnr1 + psnr2) / 2

    return psnr



