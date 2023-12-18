import itertools
from pathlib import Path
from functools import partial

import torch
import numpy as np
from skimage import io
from torch.utils import data
from torchvision.io import VideoReader
from pytorch_lightning import LightningDataModule
import torchvision.transforms.functional as F


def distortions(max_magnitude=0.5, num=6):
    distortion_fns = {"identity": lambda x: x}

    for magnitude in np.linspace(-max_magnitude, max_magnitude, num):
        distortion_fns[f"brightness{magnitude:+.2}"] = partial(F.adjust_brightness, brightness_factor=1 + magnitude)
        distortion_fns[f"contrast{magnitude:+.2}"] = partial(F.adjust_contrast, contrast_factor=1 + magnitude)
        distortion_fns[f"saturation{magnitude:+.2}"] = partial(F.adjust_saturation, saturation_factor=1 + magnitude)
        distortion_fns[f"hue{magnitude / 5:+.2}"] = partial(F.adjust_hue, hue_factor=round(magnitude / 5, 2))
        distortion_fns[f"gamma{magnitude:+.2}"] = partial(F.adjust_gamma, gamma=1 + magnitude)

    return distortion_fns


class ImageDataset(data.Dataset):
    def __init__(self, image_dir, crop_size, max_magnitude):
        self.lefts = sorted(image_dir.glob("*_L.*"))
        self.rights = sorted(image_dir.glob("*_R.*"))

        assert len(self.lefts) == len(self.rights)

        self.crop_size = crop_size

        self.distortion_fns = list(distortions(max_magnitude).values())

    def __len__(self):
        return len(self.lefts)

    @staticmethod
    def read_image(path: Path, output_size: list[int]):
        frame = torch.from_numpy(io.imread(path)).permute(2, 0, 1)
        frame = F.center_crop(frame, output_size=output_size)

        return frame

    def __getitem__(self, index):
        left = ImageDataset.read_image(self.lefts[index], self.crop_size)
        right = ImageDataset.read_image(self.rights[index], self.crop_size)

        distortion_fn = self.distortion_fns[index % len(self.distortion_fns)]
        left_distorted = distortion_fn(left)

        return {"target": left_distorted / 255, "reference": right / 255, "gt": left / 255}


class VideoDataset(data.Dataset):
    def __init__(self, video_dir, n_frames, crop_size, max_magnitude):
        self.lefts = sorted(video_dir.glob("*_L.*"))
        self.rights = sorted(video_dir.glob("*_R.*"))

        assert len(self.lefts) == len(self.rights)

        self.n_frames = n_frames
        self.crop_size = crop_size

        self.distortion_fns = list(distortions(max_magnitude).values())

    def __len__(self):
        return len(self.lefts)

    @staticmethod
    def read_video(path: Path, n_frames: int, output_size: list[int]):
        video = VideoReader(str(path), stream="video")

        frames = torch.stack([
            F.center_crop(frame["data"], output_size=output_size)
            for frame in itertools.islice(video, n_frames)
        ])

        return frames

    def __getitem__(self, index):
        left = VideoDataset.read_video(self.lefts[index], self.n_frames, self.crop_size)
        right = VideoDataset.read_video(self.rights[index], self.n_frames, self.crop_size)

        distortion_fn = self.distortion_fns[index % len(self.distortion_fns)]
        left_distorted = torch.stack([distortion_fn(frame) for frame in left])

        return {"target": left_distorted / 255, "reference": right / 255, "gt": left / 255}


class DataModule(LightningDataModule):
    def __init__(self, data_dir, n_frames, crop_size, max_magnitude, batch_size, num_workers=0):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.n_frames = n_frames
        self.crop_size = crop_size
        self.max_magnitude = max_magnitude
        self.batch_size = batch_size
        self.num_workers = num_workers

    def dataset(self, data_dir: Path):
        if self.n_frames:
            return VideoDataset(
                video_dir=data_dir,
                n_frames=self.n_frames,
                crop_size=self.crop_size,
                max_magnitude=self.max_magnitude,
            )
        else:
            return ImageDataset(
                image_dir=data_dir,
                crop_size=self.crop_size,
                max_magnitude=self.max_magnitude,
            )

    def train_dataloader(self):
        return data.DataLoader(
            dataset=self.dataset(self.data_dir / "Train"),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return data.DataLoader(
            dataset=self.dataset(self.data_dir / "Validation"),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return data.DataLoader(
            dataset=self.dataset(self.data_dir / "Test"),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
