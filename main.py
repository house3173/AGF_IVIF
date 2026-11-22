import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from decomposition.AGF_decomposition import agf_decomposition
from base_PFCFuse.base_PFCFuse import pfcfuse_fuse_gray
from detail_fusion.MVA_WLE import fuse_detail_dynamic_WLE_im2col_var_optimizer
from detail_fusion.MLE import MLE

dataset = "TNO36"
ir_folder = f".\\data\\{dataset}\\ir"
vi_folder = f".\\data\\{dataset}\\vi"
image_files = sorted(set(os.listdir(ir_folder)) & set(os.listdir(vi_folder)))
output_folder = "base_TNO36"
if not os.path.exists(f".\\output\\{output_folder}"):
    os.makedirs(f".\\output\\{output_folder}")

for number_image, image_name in enumerate(image_files, start=1):
    # Decompose image
    img_vi = cv2.imread(f"{vi_folder}\\{image_name}", cv2.IMREAD_GRAYSCALE)
    img_vi = img_vi.astype(np.float32) / 255.0
    img_ir = cv2.imread(f"{ir_folder}\\{image_name}", cv2.IMREAD_GRAYSCALE)
    img_ir = img_ir.astype(np.float32) / 255.0  

    base_vi, detail_vi = agf_decomposition(
        img_vi,
        sigma_spatial=5,
        sigma_range=0.05,
        iterations=4,
        median_ksize=3,
        numba=True
    )

    base_ir, detail_ir = agf_decomposition(
        img_ir,
        sigma_spatial=5,
        sigma_range=0.05,
        iterations=4,
        median_ksize=3,
        numba=True
    )

    base_fused = pfcfuse_fuse_gray(base_ir, base_vi)

    detail_fulsed = MLE(detail_ir, detail_vi)

    fused = np.clip(base_fused + detail_fulsed, 0, 1)

    cv2.imwrite(f".\\output\\{output_folder}\\{image_name}", (fused * 255).astype(np.uint8))

    print(f"{number_image}. Finished image {image_name}")
