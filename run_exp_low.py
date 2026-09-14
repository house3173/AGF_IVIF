import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from decomposition.AGF_decomposition import agf_decomposition
from base_PFCFuse.base_PFCFuse import pfcfuse_fuse_gray
from detail_fusion.MVA_WLE import fuse_detail_dynamic_WLE_im2col_var_optimizer
from detail_fusion.MLE import MLE

dataset = "MSRS"
number_images = 42
if dataset == "MSRS":
    number_images = 80

for low_method in ["low_1", "low_2", "low_3", "low_4", "low_5", "low_6"]:
    print()
    print("========================================")
    print(f"Processing with {low_method}...")

    output_folder = f"low_mre\\{low_method}"
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

        # base_fused = pfcfuse_fuse_gray(base_ir, base_vi)
        if low_method == "low_1":
            from compare.low.low_1 import low_1
            base_fused = low_1(base_ir, base_vi)
        elif low_method == "low_2":
            from compare.low.low_2 import low_2
            base_fused = low_2(base_ir, base_vi)
        elif low_method == "low_3":
            from compare.low.low_3 import low_3
            base_fused = low_3(base_ir, base_vi)
        elif low_method == "low_4":
            from compare.low.low_4 import low_4
            base_fused = low_4(base_ir, base_vi)
        elif low_method == "low_5":
            from compare.low.low_5 import low_5
            base_fused = low_5(base_ir, base_vi)
        elif low_method == "low_6":
            from compare.low.low_6 import low_6
            base_fused = low_6(base_ir, base_vi)

        # detail_fulsed = fuse_detail_dynamic_WLE_im2col_var_optimizer(detail_ir, detail_vi)
        detail_fulsed = MLE(detail_ir, detail_vi)

        fused = np.clip(base_fused + detail_fulsed, 0, 1)

        cv2.imwrite(f".\\data\\output\\{output_folder}\\{code_image}.png", (fused * 255).astype(np.uint8))

        print(f"{number_image}. Finished image {code_image}")
