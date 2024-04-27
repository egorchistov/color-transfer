from pathlib import Path
from functools import partial

import torch
import numpy as np
import torchvision.transforms.functional as F
from torch.utils import data
from pytorch_lightning import LightningDataModule
from torchvision.io import read_image


def setup_distortions(max_magnitude=0.5, num=6):
    distortion_fns = [lambda x: x]

    for magnitude in np.linspace(-max_magnitude, max_magnitude, num):
        distortion_fns.append(partial(F.adjust_brightness, brightness_factor=1 + magnitude))
        distortion_fns.append(partial(F.adjust_contrast, contrast_factor=1 + magnitude))
        distortion_fns.append(partial(F.adjust_saturation, saturation_factor=1 + magnitude))
        distortion_fns.append(partial(F.adjust_hue, hue_factor=magnitude))
        distortion_fns.append(partial(F.adjust_gamma, gamma=1 + magnitude))

    return distortion_fns


class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, crop_size=None):
        self.gts = sorted(image_dir.glob("*_L.*"))
        self.references = sorted(image_dir.glob("*_R.*"))

        assert len(self.gts) == len(self.references)

        self.distortion_fns = setup_distortions()
        self.crop_size = crop_size

    def __len__(self):
        return len(self.gts) * len(self.distortion_fns)

    def __getitem__(self, index):
        gt = read_image(str(self.gts[index // len(self.distortion_fns)]))
        reference = read_image(str(self.references[index // len(self.distortion_fns)]))

        if self.crop_size is not None:
            top = np.random.randint(0, gt.shape[-2] - self.crop_size[-2])
            left = np.random.randint(0, gt.shape[-1] - self.crop_size[-1])

            gt = F.crop(gt, top, left, *self.crop_size)
            reference = F.crop(reference, top, left, *self.crop_size)

            if np.random.random() > 0.5:
                # After horizontal flip left view becomes right and vice versa
                gt, reference = F.hflip(reference), F.hflip(gt)

            if np.random.random() > 0.5:
                gt, reference = F.vflip(gt), F.vflip(reference)

        distortion_fn = self.distortion_fns[index % len(self.distortion_fns)]
        target = distortion_fn(gt)

        gt, reference, target = gt / 255, reference / 255, target / 255

        if self.crop_size is not None:
            target = torch.clamp(target + 0.07 * torch.randn_like(target), min=0, max=1)
            reference = torch.clamp(reference + 0.01 * torch.randn_like(reference), min=0, max=1)

        return {"gt": gt, "reference": reference, "target": target}


class DataModule(LightningDataModule):
    def __init__(self, data_dir, crop_size, batch_size, num_workers):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=Dataset(
                image_dir=self.data_dir / "Train",
                crop_size=self.crop_size,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=Dataset(
                image_dir=self.data_dir / "Validation",
                crop_size=self.crop_size,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=Dataset(self.data_dir / "Test"),
            num_workers=self.num_workers,
        )
