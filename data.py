from pathlib import Path

import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils import data
from torchvision.utils import make_grid
from torchvision.transforms import ColorJitter, RandomCrop
from torchvision.transforms.functional import pad, crop, to_tensor
from pytorch_lightning import LightningDataModule


class Dataset(data.Dataset):
    def __init__(self, image_dir, crop_size, magnitude):
        self.crop_size = crop_size
        self.lefts = sorted(image_dir.glob("*_L.png"))
        self.rights = sorted(image_dir.glob("*_R.png"))
        self.distortions = ColorJitter(brightness=magnitude, contrast=magnitude, saturation=magnitude, hue=magnitude)

        assert len(self.lefts) == len(self.rights)

    def __len__(self):
        return len(self.lefts)

    def __getitem__(self, index):
        right = Image.open(self.rights[index]).convert("RGB")
        left_gt = Image.open(self.lefts[index]).convert("RGB")

        padding = [max(0, self.crop_size[1] - right.size[0]),
                   max(0, self.crop_size[0] - right.size[1])]

        right = pad(to_tensor(right), padding)
        left_gt = pad(to_tensor(left_gt), padding)

        crop_params = RandomCrop.get_params(right, output_size=self.crop_size)

        right = crop(right, *crop_params)
        left_gt = crop(left_gt, *crop_params)

        return self.distortions(left_gt), left_gt, right


class DataModule(LightningDataModule):
    def __init__(self, image_dir, crop_size, magnitude, batch_size, num_workers = 0):
        super().__init__()

        self.image_dir = Path(image_dir)
        self.crop_size = crop_size
        self.magnitude = magnitude
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return data.DataLoader(
            Dataset(self.image_dir / "Train", self.crop_size, self.magnitude),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=True)

    def val_dataloader(self):
        return data.DataLoader(
            Dataset(self.image_dir / "Validation", self.crop_size, self.magnitude),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=False)

    def plot_example(self):
        left, left_gt, right = next(iter(self.train_dataloader()))
        residue = (left - left_gt).abs().clamp(0, 1)
        grid = make_grid([torch.hstack(column) for column in zip(left, residue, left_gt, right)], nrow=8)

        plt.title("left, abs(left - left_gt), left_gt, right")
        plt.imshow(grid.permute(1, 2, 0))
        plt.xticks([])
        plt.yticks([])
        plt.show()


if __name__ == "__main__":
    datamodule = DataModule("Artificial Dataset", crop_size=(256, 512), magnitude=0.3, batch_size=16)
    datamodule.plot_example()
