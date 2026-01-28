import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from decomposition.AGF_decomposition import agf_decomposition
from base_PFCFuse.base_PFCFuse import pfcfuse_fuse_gray
from detail_fusion.MVA_WLE import fuse_detail_dynamic_WLE_im2col_var_optimizer

dataset = "MSRS"
number_images = 42
if dataset == "MSRS":
    number_images = 80

# for high_method in ["high_1", "high_2", "high_3", "high_4", "high_5"]:
for high_method in ["high_3", "high_4", "high_5"]:
    print()
    print("========================================")
    print(f"Processing with {high_method}...")

    output_folder = f"high\\{high_method}"
    if not os.path.exists(f".\\data\\output\\{output_folder}"):
        os.makedirs(f".\\data\\output\\{output_folder}")

    for i in range(number_images):
        number_image = i + 1
        if number_image < 10:
            code_image = '0'+str(number_image)
        else:
            code_image = str(number_image)

        # Decompose image
        img_vi = cv2.imread(f".\\data\\{dataset}\\vi\\{code_image}.png", cv2.IMREAD_GRAYSCALE)
        img_vi = img_vi.astype(np.float32) / 255.0
        img_ir = cv2.imread(f".\\data\\{dataset}\\ir\\{code_image}.png", cv2.IMREAD_GRAYSCALE)
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

        if high_method == "high_1":
            from compare.high.high_1 import high_1
            detail_fused = high_1(detail_ir, detail_vi)
        elif high_method == "high_2":
            from compare.high.high_2 import high_2
            detail_fused = high_2(detail_ir, detail_vi)
        elif high_method == "high_3":
            from compare.high.high_3 import high_3
            detail_fused = high_3(detail_ir, detail_vi)
        elif high_method == "high_4":
            from compare.high.high_4 import high_4
            detail_fused = high_4(detail_ir, detail_vi)
        elif high_method == "high_5":
            from compare.high.high_5 import high_5
            detail_fused = high_5(detail_ir, detail_vi)

        base_fused = pfcfuse_fuse_gray(base_ir, base_vi)

        fused = np.clip(base_fused + detail_fused, 0, 1)

        cv2.imwrite(f".\\data\\output\\{output_folder}\\{code_image}.png", (fused * 255).astype(np.uint8))

        print(f"{number_image}. Finished image {code_image}")
