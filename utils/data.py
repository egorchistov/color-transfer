from pathlib import Path
from functools import partial

import torch
import numpy as np
from skimage import io
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import torchvision.transforms.functional as F  # noqa


def get_distortions(max_magnitude=0.5, num=6):
    assert max_magnitude <= 0.5

    distortion_fns = [lambda x: x]

    for magnitude in np.linspace(-max_magnitude, max_magnitude, num):
        distortion_fns.extend([
            partial(F.adjust_brightness, brightness_factor=1 + magnitude),
            partial(F.adjust_contrast, contrast_factor=1 + magnitude),
            partial(F.adjust_saturation, saturation_factor=1 + magnitude),
            partial(F.adjust_hue, hue_factor=magnitude / 5),
            partial(F.adjust_gamma, gamma=1 + magnitude)
        ])

    return distortion_fns


class PairDataset(Dataset):
    def __init__(self, image_dir):
        self.gts = sorted(image_dir.glob("*_L.*"))
        self.references = sorted(image_dir.glob("*_R.*"))

        assert len(self.gts) == len(self.references)

    def __len__(self):
        return len(self.gts)

    @staticmethod
    def read_image(path: Path):
        return torch.from_numpy(io.imread(path)).permute(2, 0, 1)

    def __getitem__(self, index):
        gt = PairDataset.read_image(self.gts[index]) / 255
        reference = PairDataset.read_image(self.references[index]) / 255

        return {"gt": gt, "reference": reference}


class TestDataset(PairDataset):
    def __init__(self, image_dir):
        super().__init__(image_dir)

        self.distortion_fns = get_distortions()

    def __len__(self):
        return len(self.distortion_fns) * super().__len__()

    def __getitem__(self, index):
        return_dict = super().__getitem__(index // len(self.distortion_fns))
        distortion_fn = self.distortion_fns[index % len(self.distortion_fns)]
        return_dict["target"] = distortion_fn(return_dict["gt"])

        return return_dict


class DataModule(LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers=0):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

    @staticmethod
    def collate_fn(batch):
        min_height = min(item["gt"].shape[-2] for item in batch)
        min_width = min(item["gt"].shape[-1] for item in batch)

        return {
            k: torch.stack([item[k][:, :min_height, :min_width] for item in batch])
            for k in batch[0].keys()
        }

    def train_dataloader(self):
        dataset = PairDataset(self.data_dir / "Train")

        return DataLoader(dataset, self.batch_size, shuffle=True, num_workers=self.num_workers,
                          collate_fn=DataModule.collate_fn)

    def test_dataloader(self):
        dataset = TestDataset(self.data_dir / "Test")

        return DataLoader(dataset, self.batch_size, num_workers=self.num_workers,
                          collate_fn=DataModule.collate_fn)
