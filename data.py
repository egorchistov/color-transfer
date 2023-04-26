from pathlib import Path

import torch
import torchvision.utils
import pytorch_lightning as pl
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from PIL import Image
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


distortions = A.OneOf([
    A.RandomBrightnessContrast(contrast_limit=0, brightness_limit=(-0.3, -0.1)),
    A.RandomBrightnessContrast(contrast_limit=0, brightness_limit=(0.1, 0.3)),
    A.RandomBrightnessContrast(contrast_limit=(-0.3, -0.1), brightness_limit=0),
    A.RandomBrightnessContrast(contrast_limit=(0.1, 0.3), brightness_limit=0),
    A.RandomGamma(gamma_limit=(70, 90)),
    A.RandomGamma(gamma_limit=(110, 130)),
    A.HueSaturationValue(hue_shift_limit=(-30, -10), sat_shift_limit=0, val_shift_limit=0),
    A.HueSaturationValue(hue_shift_limit=(10, 30), sat_shift_limit=0, val_shift_limit=0),
    A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=(-30, -10), val_shift_limit=0),
    A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=(10, 30), val_shift_limit=0),
    A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=0, val_shift_limit=(-30, -10)),
    A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=0, val_shift_limit=(10, 30)),
], p=36 / 37)


class CTDataset(Dataset):
    def __init__(self, image_dir: Path, transforms, distortions, use_real_distortions):
        """First distortions are applied to the left image, then transforms are applied to both"""

        self.lefts = sorted(image_dir.glob("*_L.png"))
        self.lefts_distorted = sorted(image_dir.glob("*_LD.png"))
        self.rights = sorted(image_dir.glob("*_R.png"))

        assert len(self.lefts) == len(self.rights)

        self.transforms = transforms
        self.distortions = distortions
        self.use_real_distortions = use_real_distortions

    def __len__(self):
        return len(self.lefts)

    def __getitem__(self, index):
        left_gt = np.array(Image.open(self.lefts[index]).convert("RGB"))
        right = np.array(Image.open(self.rights[index]).convert("RGB"))

        if self.use_real_distortions:
            left = np.array(Image.open(self.lefts_distorted[index]).convert("RGB"))
        else:
            left = left_gt

        left = self.distortions(image=left)["image"]

        t = self.transforms(image=left, left_gt=left_gt, right=right)
        left, left_gt, right = t["image"], t["left_gt"], t["right"]

        return left, left_gt, right


class CTDataModule(pl.LightningDataModule):
    def __init__(self, image_dir, batch_size, patch_size, num_workers, use_real_distortions):
        super().__init__()

        self.image_dir = image_dir
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.use_real_distortions = use_real_distortions

        self.train = None
        self.val = None
        self.pred = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_transforms = A.Compose([
                A.PadIfNeeded(self.patch_size[0], self.patch_size[1]),
                A.RandomCrop(self.patch_size[0], self.patch_size[1]),
                A.ToFloat(),
                ToTensorV2()
            ], additional_targets={"left_gt": "image", "right": "image"})

            self.train = CTDataset(self.image_dir / "Train", train_transforms, distortions, self.use_real_distortions)

            val_transforms = A.Compose([
                A.PadIfNeeded(self.patch_size[0], self.patch_size[1]),
                A.CenterCrop(self.patch_size[0], self.patch_size[1]),
                A.ToFloat(),
                ToTensorV2()
            ], additional_targets={"left_gt": "image", "right": "image"})

            self.val = CTDataset(self.image_dir / "Validation", val_transforms, distortions, self.use_real_distortions)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def plot_example(self):
        idx = np.random.choice(len(self.train))
        left, left_gt, right = self.train[idx]
        residue = (left - left_gt).abs().clamp(0, 1)
        grid = torchvision.utils.make_grid([left, residue, left_gt, right], nrow=2)

        plt.figure(figsize=(8, 5))
        plt.title("Left, |Left - Left GT|,\nLeft GT, Right")
        plt.imshow(grid.permute(1, 2, 0))
        plt.xticks([])
        plt.yticks([])
        plt.show()


if __name__ == "__main__":
    datamodule = CTDataModule(Path("Artificial Dataset"), batch_size=1, patch_size=(256, 512), num_workers=0,
                              use_real_distortions=False)
    datamodule.setup()
    datamodule.plot_example()
