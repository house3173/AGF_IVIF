import pandas as pd
from metric.eval_one_image import evaluation_one

# Danh sách lưu kết quả từng ảnh
dataset = "MSRS"
root_folder = f".\\data\\output\\high"

number_images = 42
if dataset == "MSRS":
    number_images = 80

# for method in ["low_1", "low_2", "low_3", "low_4", "low_5", "low_6"]:
for method in ["high_1", "high_2", "high_3", "high_4", "high_5"]:
    results = []
    for i in range(number_images):
        number_image = i + 1
        code_image = f"{number_image:02d}"

        ir_image = f".\\data\\{dataset}\\vi\\{code_image}.png"
        vi_image = f".\\data\\{dataset}\\ir\\{code_image}.png"
        fused_image = f"{root_folder}\\{method}\\{code_image}.png"

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
