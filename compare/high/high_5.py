import cv2
import numpy as np

def morphological_gradient(img, se_radius, levels):
    """ Tính toán Gradient hình thái đa mức (MLMG) """
    mlmg = np.zeros_like(img, dtype=np.float32)
    weights = [1 / (2 * i + 1) for i in range(1, levels + 1)]
    
    for i in range(1, levels + 1):
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_radius * i, se_radius * i))
        dilated = cv2.dilate(img, se)
        eroded = cv2.erode(img, se)
        gradient = dilated - eroded
        mlmg += weights[i - 1] * gradient
    
    return mlmg

def area_energy(img):
    """ Tính toán năng lượng vùng (AE) """
    kernel = np.array([[1, 3, 1], [2, 6, 2], [1, 3, 1]]) / 20.0
    ae = cv2.filter2D(img ** 2, -1, kernel)
    return ae

def neighborhood_energy(detail_layer, N=1):
    """ Tính toán năng lượng lân cận (NE) """
    kernel = np.ones((2 * N + 1, 2 * N + 1), dtype=np.float32)
    return cv2.filter2D(detail_layer, -1, kernel)

def enhanced_detail_layer(detail_layer, N=1):
    """ Tính toán lớp chi tiết tăng cường DD_n """
    NE_D = neighborhood_energy(detail_layer, N)
    return NE_D * detail_layer * detail_layer

def high_5(detail_ir, detail_vi):
    """ Hợp nhất hai thành phần tần số cao theo mc-DTNP """
    # Tính MLMG và AE cho cả hai lớp chi tiết
    mlmg_1 = morphological_gradient(detail_ir, se_radius=3, levels=3)
    mlmg_2 = morphological_gradient(detail_vi, se_radius=3, levels=3)
    
    ae_1 = area_energy(detail_ir)
    ae_2 = area_energy(detail_vi)
    
    # Tính lớp chi tiết tăng cường
    dd_1 = enhanced_detail_layer(detail_ir)
    dd_2 = enhanced_detail_layer(detail_vi)
    
    # Tính toán năng lượng tổng hợp
    E1 = np.maximum(mlmg_1, np.maximum(ae_1, dd_1))
    E2 = np.maximum(mlmg_2, np.maximum(ae_2, dd_2))
    
    # Xác định ma trận quyết định hợp nhất
    Matrice = np.where(E1 >= E2, 1, 0)
    
    # Tạo ảnh hợp nhất
    detail_fused = Matrice * detail_ir + (1 - Matrice) * detail_vi
    
    return detail_fused