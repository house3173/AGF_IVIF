from base_fusion.training_cosinelr_earlystop import train_fusion

if __name__ == "__main__":
    train_fusion(
        root_dir="/u01/data/smart_filter/hautv/AGF_IVIF/data/training_model",
        folder_train_outpath="/u01/data/smart_filter/hautv/AGF_IVIF/base_fusion/run/train_4",
        epochs=100,
        batch_size=2,
        lr=1e-4,
        device="cuda"
    )