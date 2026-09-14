import numpy as np
from scipy.ndimage import convolve1d
from scipy.ndimage import gaussian_filter

def gauss_filter(s, sigma):
    """
    Equivalent to MATLAB's:
    ksize=bitor(round(5*sigma),1);
    g=fspecial('gaussian',[1,ksize],sigma);
    t=imfilter(s,g,'replicate');
    Note: MATLAB [1, ksize] kernel blurs horizontally (along axis 1).
    """
    # Calculate kernel size (ensure it's odd)
    ksize = int(round(5 * sigma)) | 1
    
    # Create 1D Gaussian kernel
    x = np.arange(-(ksize // 2), (ksize // 2) + 1)
    g = np.exp(-(x**2) / (2 * sigma**2))
    g = g / np.sum(g)  # Normalize
    
    # Apply filter along axis 1 (horizontal), equivalent to [1, ksize] in MATLAB
    # mode='nearest' matches MATLAB's 'replicate'
    if s.ndim == 3:
        t = np.zeros_like(s)
        for c in range(s.shape[2]):
            t[:, :, c] = convolve1d(s[:, :, c], g, axis=1, mode='nearest')
    else:
        t = convolve1d(s, g, axis=1, mode='nearest')
        
    return t

def imgaussfilt_py(img, sigma=0.5, truncate=2.0):
    """
    Python equivalent of MATLAB imgaussfilt
    - sigma: standard deviation (pixel domain)
    - truncate: 2.0 -> equivalent to MATLAB (±2σ)
    """
    return gaussian_filter(
        img,
        sigma=sigma,
        mode='nearest',   # replicate padding
        truncate=truncate
    )