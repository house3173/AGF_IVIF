import os
import pandas as pd
from metric.eval_one_image import evaluation_one

# Danh sách lưu kết quả từng ảnh
dataset = "TNO36"
root_folder = f".\\output"
ir_folder = f".\\data\\{dataset}\\ir"
vi_folder = f".\\data\\{dataset}\\vi"
image_files = sorted(set(os.listdir(ir_folder)) & set(os.listdir(vi_folder)))

# for method in ["low_1", "low_2", "low_3", "low_4", "low_5", "low_6"]:
for method in ["improve_TNO36"]:
    results = []
    for image_name in image_files:
        code_image = os.path.splitext(image_name)[0]

        ir_image = f"{vi_folder}\\{image_name}"
        vi_image = f"{ir_folder}\\{image_name}"
        fused_image = f"{root_folder}\\{method}\\{image_name}"

        EN, MI, SF, AG, SD, MLI, CC, SCD, VIF, MSE, PSNR, Qabf, Nabf, SSIM, MS_SSIM = \
            evaluation_one(ir_image, vi_image, fused_image)

        results.append({
            "Image": code_image,
            "EN": EN,
            "MI": MI,
            "SF": SF,
            "AG": AG,
            "SD": SD,
            "MLI": MLI,
            "CC": CC,
            "SCD": SCD,
            "VIF": VIF,
            "MSE": MSE,
            "PSNR": PSNR,
            "Qabf": Qabf,
            "Nabf": Nabf,
            "SSIM": SSIM,
            "MS_SSIM": MS_SSIM
        })

    # Tạo DataFrame
    df = pd.DataFrame(results)

    # Làm tròn cho đẹp
    df = df.round(6)

    # Tính trung bình
    mean_row = df.drop(columns=["Image"]).mean()
    mean_row["Image"] = "Mean"

    # Ghép dòng Mean vào cuối bảng
    df = pd.concat([df, mean_row.to_frame().T], ignore_index=True)

    # Lưu ra Excel
    output_excel = f"{root_folder}\\{method}\\metrics_{dataset}.xlsx"
    df.to_excel(output_excel, index=False)

    print(f"✓ Saved metrics to {output_excel}")
    print(df)
