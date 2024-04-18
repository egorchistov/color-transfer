from pathlib import Path
from functools import partial

import torch
import numpy as np
from skimage import io
from torch.utils import data
from pytorch_lightning import LightningDataModule
import torchvision.transforms.functional as F


def setup_distortions(max_magnitude=0.5, num=6):
    distortion_fns = [lambda x: x]

    for magnitude in np.linspace(-max_magnitude, max_magnitude, num):
        distortion_fns.append(partial(F.adjust_brightness, brightness_factor=1 + magnitude))
        distortion_fns.append(partial(F.adjust_contrast, contrast_factor=1 + magnitude))
        distortion_fns.append(partial(F.adjust_saturation, saturation_factor=1 + magnitude))
        distortion_fns.append(partial(F.adjust_hue, hue_factor=round(magnitude / 5, 2)))
        distortion_fns.append(partial(F.adjust_gamma, gamma=1 + magnitude))

    return distortion_fns


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, crop_size=None):
        self.gts = sorted(image_dir.glob("*_L.*"))
        self.references = sorted(image_dir.glob("*_R.*"))

        assert len(self.gts) == len(self.references)

        self.crop_size = crop_size

    def __len__(self):
        return len(self.gts)

    @staticmethod
    def read_image(path: Path, output_size=None):
        frame = torch.from_numpy(io.imread(path)).permute(2, 0, 1)

        if output_size is not None:
            frame = F.center_crop(frame, output_size=output_size)

        return frame

    def __getitem__(self, index):
        gt = PairDataset.read_image(self.gts[index], self.crop_size)
        reference = PairDataset.read_image(self.references[index], self.crop_size)

        return {"gt": gt, "reference": reference}


class TrainValDataset(PairDataset):
    def __init__(self, image_dir, crop_size, max_magnitude):
        super().__init__(image_dir, crop_size)

        self.distortion_fns = setup_distortions(max_magnitude)

    def __getitem__(self, index):
        return_dict = super().__getitem__(index)
        distortion_fn = self.distortion_fns[index % len(self.distortion_fns)]
        return_dict["target"] = distortion_fn(return_dict["gt"])

        return {k: v / 255 for k, v in return_dict.items()}


class TestDataset(PairDataset):
    def __init__(self, image_dir):
        super().__init__(image_dir)

        self.distortion_fns = setup_distortions()

    def __len__(self):
        return super().__len__() * len(self.distortion_fns)

    def __getitem__(self, index):
        return_dict = super().__getitem__(index // len(self.distortion_fns))
        distortion_fn = self.distortion_fns[index % len(self.distortion_fns)]
        return_dict["target"] = distortion_fn(return_dict["gt"])

        return {k: v / 255 for k, v in return_dict.items()}


class DataModule(LightningDataModule):
    def __init__(self, data_dir, crop_size, max_magnitude, batch_size, num_workers):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.crop_size = crop_size
        self.max_magnitude = max_magnitude
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=TrainValDataset(
                image_dir=self.data_dir / "Train",
                crop_size=self.crop_size,
                max_magnitude=self.max_magnitude,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=TrainValDataset(
                image_dir=self.data_dir / "Validation",
                crop_size=self.crop_size,
                max_magnitude=self.max_magnitude,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=TestDataset(self.data_dir / "Test"),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
