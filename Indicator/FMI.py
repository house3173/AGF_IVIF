import numpy as np
from scipy import ndimage
from Indicator.utils import *

def rerange(im):
    """Chuẩn hóa ảnh về khoảng [0, 1]"""
    im = im.astype(np.float64)
    im_min = im.min()
    im_max = im.max()
    if im_max == im_min:
        return np.ones_like(im)
    return (im - im_min) / (im_max - im_min)

def calculate_gradient(img):
    """Tính gradient sử dụng Sobel operator"""
    dx = ndimage.sobel(img, axis=0, mode='constant')
    dy = ndimage.sobel(img, axis=1, mode='constant')
    return np.sqrt(dx**2 + dy**2)

def fmi(ima, imb, imf, w=3):

    ima = convert_to_grayscale(ima)
    imb = convert_to_grayscale(imb)
    imf = convert_to_grayscale(imf)

    # Kiểm tra kích thước ảnh
    if ima.shape != imb.shape or ima.shape != imf.shape:
        raise ValueError("Tất cả ảnh phải có cùng kích thước")
    
    # Chuyển đổi sang float64
    ima = ima.astype(np.float64)
    imb = imb.astype(np.float64)
    imf = imf.astype(np.float64)

    # Trích xuất đặc trưng gradient
    aFeature = calculate_gradient(ima)
    bFeature = calculate_gradient(imb)
    fFeature = calculate_gradient(imf)
    
    # Thiết lập cửa sổ trượt
    w_half = w // 2
    rows, cols = aFeature.shape
    fmi_map = np.zeros((rows - 2*w_half, cols - 2*w_half))
    
    for p in range(w_half, rows - w_half):
        for q in range(w_half, cols - w_half):
            # Trích cửa sổ
            aSub = aFeature[p-w_half:p+w_half+1, q-w_half:q+w_half+1]
            bSub = bFeature[p-w_half:p+w_half+1, q-w_half:q+w_half+1]
            fSub = fFeature[p-w_half:p+w_half+1, q-w_half:q+w_half+1]
            
            # Tính FMI cho a và f
            if np.array_equal(aSub, fSub):
                fmi_af = 1.0
            else:
                aSub_norm = rerange(aSub)
                fSub_norm = rerange(fSub)
                
                # Tính PDF
                aPdf = aSub_norm / np.sum(aSub_norm)
                fPdf = fSub_norm / np.sum(fSub_norm)
                
                # Tính CDF
                aCdf = np.cumsum(aPdf.flatten())
                fCdf = np.cumsum(fPdf.flatten())
                
                # Tính hệ số tương quan Pearson
                a_mean = np.mean(aPdf)
                f_mean = np.mean(fPdf)
                cov = np.sum((aPdf - a_mean) * (fPdf - f_mean))
                var_a = np.sum((aPdf - a_mean)**2)
                var_f = np.sum((fPdf - f_mean)**2)
                c = cov / np.sqrt(var_a * var_f) if (var_a * var_f) != 0 else 0
                
                # Tính entropy và mutual information (đơn giản hóa)
                epsilon = 1e-10
                a_entropy = -np.sum(aPdf * np.log2(aPdf + 1e-10))
                f_entropy = -np.sum(fPdf * np.log2(fPdf + 1e-10))
                
                # Joint entropy approximation
                joint_pdf = aPdf * fPdf
                joint_entropy = -np.sum(joint_pdf * np.log2(joint_pdf + 1e-10))
                
                mi = a_entropy + f_entropy - joint_entropy
                fmi_af = 2 * mi / (a_entropy + f_entropy) if (a_entropy + f_entropy) != 0 else 0
            
            # Tính FMI cho b và f (tương tự)
            if np.array_equal(bSub, fSub):
                fmi_bf = 1.0
            else:
                bSub_norm = rerange(bSub)
                fSub_norm = rerange(fSub)
                
                bPdf = bSub_norm / np.sum(bSub_norm)
                fPdf = fSub_norm / np.sum(fSub_norm)
                
                b_entropy = -np.sum(bPdf * np.log2(bPdf + 1e-10))
                f_entropy = -np.sum(fPdf * np.log2(fPdf + 1e-10))
                
                joint_pdf = bPdf * fPdf
                joint_entropy = -np.sum(joint_pdf * np.log2(joint_pdf + 1e-10))
                
                mi = b_entropy + f_entropy - joint_entropy
                fmi_bf = 2 * mi / (b_entropy + f_entropy) if (b_entropy + f_entropy) != 0 else 0
            
            fmi_map[p-w_half, q-w_half] = (fmi_af + fmi_bf) / 2
    
    return np.mean(fmi_map)