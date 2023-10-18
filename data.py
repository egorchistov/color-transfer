import random
import itertools
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from torch.utils import data
from torchvision.io import VideoReader
from torchvision.utils import make_grid
from torchvision.transforms import ColorJitter, RandomCrop
from torchvision.transforms.functional import crop
from pytorch_lightning import LightningDataModule

from methods.visualizations import chess_mix, rgbmse, rgbssim


class Dataset(data.Dataset):
    def __init__(self, video_dir, n_frames, crop_size, magnitude):
        self.lefts = sorted(video_dir.glob("*_L.mp4"))
        self.rights = sorted(video_dir.glob("*_R.mp4"))

        assert len(self.lefts) == len(self.rights)

        self.n_frames = n_frames
        self.crop_size = crop_size

        self.magnitude = magnitude

    def __len__(self):
        return len(self.lefts)

    def read_video(self, src: Path, start=None, crop_params=None):
        video = VideoReader(str(src), stream="video")

        if start is None:
            metadata = video.get_metadata()

            max_start = int(metadata["video"]["duration"][0] * metadata["video"]["fps"][0]) - self.n_frames
            start = random.randint(0, max_start)

        frames = []
        for frame in itertools.islice(video, start, start + self.n_frames):
            if crop_params is None:
                crop_params = RandomCrop.get_params(frame["data"], output_size=self.crop_size)

            frames.append(crop(frame["data"], *crop_params))

        return torch.stack(frames), start, crop_params

    def __getitem__(self, index):
        left_gt, start, crop_params = self.read_video(self.lefts[index])
        right, _, _ = self.read_video(self.rights[index], start, crop_params)

        distortion_params = ColorJitter.get_params(
            brightness=[1 - self.magnitude, 1 + self.magnitude],
            contrast=[1 - self.magnitude, 1 + self.magnitude],
            saturation=[1 - self.magnitude, 1 + self.magnitude],
            hue=[-self.magnitude, self.magnitude])

        distortion = ColorJitter()
        distortion.get_params = lambda *_: distortion_params

        left = torch.stack([distortion(frame) for frame in left_gt])

        return left / 255, left_gt / 255, right / 255


class DataModule(LightningDataModule):
    def __init__(self, video_dir, n_frames, crop_size, magnitude, batch_size, num_workers=0):
        super().__init__()

        self.video_dir = Path(video_dir)
        self.n_frames = n_frames
        self.crop_size = crop_size
        self.magnitude = magnitude
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return data.DataLoader(
            Dataset(self.video_dir / "Train", self.n_frames, self.crop_size, self.magnitude),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True)

    def val_dataloader(self):
        return data.DataLoader(
            Dataset(self.video_dir / "Validation", self.n_frames, self.crop_size, self.magnitude),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False)

    def plot_example(self):
        left, left_gt, right = (view.flatten(end_dim=1)
                                for view in next(iter(self.train_dataloader())))

        grid = make_grid(torch.cat([
            chess_mix(left, left_gt),
            rgbmse(left, left_gt),
            rgbssim(left, left_gt),
            right
        ], dim=-1), nrow=1)

        plt.title("Left Ground Truth/Distorted, RGB MSE Error, RGB SSIM Error, Right")
        plt.imshow(grid.permute(1, 2, 0))
        plt.xticks([])
        plt.yticks([])
        plt.show()


if __name__ == "__main__":
    datamodule = DataModule("3DMovies", n_frames=1, crop_size=(512, 512), magnitude=0.3, batch_size=7)
    datamodule.plot_example()
