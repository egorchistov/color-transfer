import itertools
from functools import partial
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torch.utils import data
from torchvision.io import VideoReader
from torchvision.utils import make_grid
from pytorch_lightning import LightningDataModule

from methods.visualizations import chess_mix, rgbmse, rgbssim


class Dataset(data.Dataset):
    def __init__(self, video_dir, n_frames, crop_size, magnitude):
        self.lefts = sorted(video_dir.glob("*_L.mp4"))
        self.rights = sorted(video_dir.glob("*_R.mp4"))

        assert len(self.lefts) == len(self.rights)

        self.n_frames = n_frames
        self.crop_size = crop_size

        self.distortion_fns = []
        for magnitude in np.linspace(-magnitude, magnitude, num=6):
            self.distortion_fns.extend([
                partial(F.adjust_brightness, brightness_factor=1 + magnitude),
                partial(F.adjust_contrast, contrast_factor=1 + magnitude),
                partial(F.adjust_saturation, saturation_factor=1 + magnitude),
                partial(F.adjust_hue, hue_factor=magnitude),
            ])

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
        left = Dataset.read_video(self.lefts[index], self.n_frames, self.crop_size)
        right = Dataset.read_video(self.rights[index], self.n_frames, self.crop_size)

        distortion_fn = self.distortion_fns[index % len(self.distortion_fns)]
        left_distorted = torch.stack([distortion_fn(frame) for frame in left])

        return left_distorted / 255, left / 255, right / 255


class DataModule(LightningDataModule):
    def __init__(self, video_dir, n_frames, crop_size, magnitude, batch_size, num_workers=0):
        super().__init__()

        self.video_dir = Path(video_dir)
        self.n_frames = n_frames
        self.crop_size = crop_size
        self.magnitude = magnitude
        self.batch_size = batch_size
        self.num_workers = num_workers

    def unified_dataset(self, video_dir: Path):
        return Dataset(
            video_dir=video_dir,
            n_frames=self.n_frames,
            crop_size=self.crop_size,
            magnitude=self.magnitude,
        )

    def unified_dataloader(self, video_dir: Path):
        return data.DataLoader(
            dataset=self.unified_dataset(video_dir),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def train_dataloader(self):
        return self.unified_dataloader(video_dir=self.video_dir / "Train")

    def val_dataloader(self):
        return self.unified_dataloader(video_dir=self.video_dir / "Validation")

    def plot_example(self):
        left, left_gt, right = (view.flatten(end_dim=1)
                                for view in next(iter(self.val_dataloader())))

        grid = make_grid(torch.cat([
            chess_mix(left, left_gt),
            rgbmse(left, left_gt),
            rgbssim(left, left_gt),
        ], dim=-1), nrow=1)

        plt.title("Left Ground Truth/Distorted, RGB MSE Error, RGB SSIM Error")
        plt.imshow(grid.permute(1, 2, 0))
        plt.xticks([])
        plt.yticks([])
        plt.show()


if __name__ == "__main__":
    datamodule = DataModule("3DMovies", n_frames=1, crop_size=(512, 1024), magnitude=0.3, batch_size=8)
    datamodule.plot_example()
