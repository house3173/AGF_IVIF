import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class FusionDataset(Dataset):
    def __init__(self, root_dir, mode="train", transform=None):
        """
        root_dir/
            train/
                ir_base/
                vi_base/
            valid/
                ir_base/
                vi_base/
        """
        self.ir_dir = os.path.join(root_dir, mode, "ir_base")
        self.vi_dir = os.path.join(root_dir, mode, "vi_base")

        self.filenames = sorted(os.listdir(self.ir_dir))
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]

        ir_path = os.path.join(self.ir_dir, name)
        vi_path = os.path.join(self.vi_dir, name)

        ir = Image.open(ir_path).convert("L")
        vi = Image.open(vi_path).convert("L")

        if self.transform:
            ir = self.transform(ir)
            vi = self.transform(vi)

        return {
            "ir": ir,
            "vi": vi,
            "name": name
        }

def get_dataloader(root_dir, mode, batch_size, num_workers=4):
    transform = T.Compose([
        T.ToTensor(),   # [0,1], shape (1,H,W)
    ])

    dataset = FusionDataset(
        root_dir=root_dir,
        mode=mode,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(mode == "train")
    )
    return loader