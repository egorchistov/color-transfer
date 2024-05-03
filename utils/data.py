from pathlib import Path
from functools import partial

import torch
import numpy as np
import torchvision.transforms.functional as F
from torch.utils import data
from pytorch_lightning import LightningDataModule
from torchvision.io import read_image


def setup_grid_distortions(max_magnitude=0.5, num=6):
    distortion_fns = [lambda x: x]

    for magnitude in np.linspace(-max_magnitude, max_magnitude, num):
        distortion_fns.append(partial(F.adjust_brightness, brightness_factor=1 + magnitude))
        distortion_fns.append(partial(F.adjust_contrast, contrast_factor=1 + magnitude))
        distortion_fns.append(partial(F.adjust_saturation, saturation_factor=1 + magnitude))
        distortion_fns.append(partial(F.adjust_hue, hue_factor=magnitude))
        distortion_fns.append(partial(F.adjust_gamma, gamma=1 + magnitude))

    return distortion_fns


def apply_uniform_distortions(img, max_magnitude=0.5):
    fn_idx = torch.randperm(6)

    brightness_factor = np.random.uniform(1 - max_magnitude, 1 + max_magnitude)
    contrast_factor = np.random.uniform(1 - max_magnitude, 1 + max_magnitude)
    saturation_factor = np.random.uniform(1 - max_magnitude, 1 + max_magnitude)
    hue_factor = np.random.uniform(-max_magnitude, max_magnitude)
    gamma = np.random.uniform(1 - max_magnitude, 1 + max_magnitude)
    sharpness_factor = np.random.uniform(1 - max_magnitude, 1 + max_magnitude)

    for fn_id in fn_idx:
        if fn_id == 0:
            img = F.adjust_brightness(img, brightness_factor)
        elif fn_id == 1:
            img = F.adjust_contrast(img, contrast_factor)
        elif fn_id == 2:
            img = F.adjust_saturation(img, saturation_factor)
        elif fn_id == 3:
            img = F.adjust_hue(img, hue_factor)
        elif fn_id == 4:
            img = F.adjust_gamma(img, gamma)
        elif fn_id == 5:
            img = F.adjust_sharpness(img, sharpness_factor)

    return img


class ArtificialTrainValDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, crop_size, image_repeats):
        self.gts = sorted(image_dir.glob("*_L.*"))
        self.references = sorted(image_dir.glob("*_R.*"))

        assert len(self.gts) == len(self.references)

        self.crop_size = crop_size
        self.image_repeats = image_repeats

    def __len__(self):
        return len(self.gts) * self.image_repeats

    def __getitem__(self, index):
        gt = read_image(str(self.gts[index // self.image_repeats]))
        reference = read_image(str(self.references[index // self.image_repeats]))

        top = np.random.randint(0, gt.shape[-2] - self.crop_size[-2])
        left = np.random.randint(0, gt.shape[-1] - self.crop_size[-1])

        gt = F.crop(gt, top, left, *self.crop_size)
        reference = F.crop(reference, top, left, *self.crop_size)

        if np.random.random() > 0.5:
            # After horizontal flip left view becomes right view and vice versa
            gt, reference = F.hflip(reference), F.hflip(gt)

        if np.random.random() > 0.5:
            gt, reference = F.vflip(gt), F.vflip(reference)

        target = apply_uniform_distortions(gt)

        return {"gt": gt / 255, "reference": reference / 255, "target": target / 255}


class ArtificialTestDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir):
        self.gts = sorted(image_dir.glob("*_L.*"))
        self.references = sorted(image_dir.glob("*_R.*"))

        assert len(self.gts) == len(self.references)

        self.distortion_fns = setup_grid_distortions()

    def __len__(self):
        return len(self.gts) * len(self.distortion_fns)

    def __getitem__(self, index):
        gt = read_image(str(self.gts[index // len(self.distortion_fns)]))
        reference = read_image(str(self.references[index // len(self.distortion_fns)]))

        distortion_fn = self.distortion_fns[index % len(self.distortion_fns)]
        target = distortion_fn(gt)

        return {"gt": gt / 255, "reference": reference / 255, "target": target / 255}


class RealWorldTestDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir):
        self.gts = sorted(image_dir.glob("*/*_L.*"))
        self.targets = sorted(image_dir.glob("*/*_LD.*"))
        self.references = sorted(image_dir.glob("*/*_R.*"))

        assert len(self.gts) == len(self.targets) == len(self.references)

    def __len__(self):
        return len(self.gts)

    def __getitem__(self, index):
        gt = read_image(str(self.gts[index]))
        target = read_image(str(self.targets[index]))
        reference = read_image(str(self.references[index]))

        return {"gt": gt / 255, "reference": reference / 255, "target": target / 255}


class DataModule(LightningDataModule):
    def __init__(self, data_dir, crop_size, image_repeats, batch_size, num_workers):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.crop_size = crop_size
        self.image_repeats = image_repeats
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=ArtificialTrainValDataset(
                image_dir=self.data_dir / "Train",
                crop_size=self.crop_size,
                image_repeats=self.image_repeats,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        artificial_dataloader = torch.utils.data.DataLoader(
            dataset=ArtificialTrainValDataset(
                image_dir=self.data_dir / "Validation",
                crop_size=self.crop_size,
                image_repeats=self.image_repeats,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        realworld_dataloader = torch.utils.data.DataLoader(
            dataset=RealWorldTestDataset(self.data_dir / "Real-World Test"),
            num_workers=self.num_workers,
        )

        return [artificial_dataloader, realworld_dataloader]

    def test_dataloader(self):
        artificial_dataloader = torch.utils.data.DataLoader(
            dataset=ArtificialTestDataset(self.data_dir / "Test"),
            num_workers=self.num_workers,
        )

        realworld_dataloader = torch.utils.data.DataLoader(
            dataset=RealWorldTestDataset(self.data_dir / "Real-World Test"),
            num_workers=self.num_workers,
        )

        return [artificial_dataloader, realworld_dataloader]
