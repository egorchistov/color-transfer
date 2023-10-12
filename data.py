import itertools
import random
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from torch.utils import data
from torchvision.io import VideoReader
from torchvision.utils import make_grid
from torchvision.transforms import ColorJitter, RandomCrop
from torchvision.transforms.functional import crop
from pytorch_lightning import LightningDataModule


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

    def read_video(self, src: Path):
        video = VideoReader(str(src), stream="video")

        metadata = video.get_metadata()
        max_start = int(metadata["video"]["duration"][0] * metadata["video"]["fps"][0]) - self.n_frames

        start = random.randint(0, max_start)

        frames = torch.stack([
            frame["data"]
            for frame in itertools.islice(video, start, start + self.n_frames)
        ])

        return frames

    def __getitem__(self, index):
        left_gt = self.read_video(self.lefts[index])
        right = self.read_video(self.rights[index])

        crop_params = RandomCrop.get_params(right, output_size=self.crop_size)
        left_gt = torch.stack([crop(frame, *crop_params) for frame in left_gt])
        right = torch.stack([crop(frame, *crop_params) for frame in right])

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
        left, left_gt, right = next(iter(self.train_dataloader()))
        left, left_gt, right = left.flatten(end_dim=1), left_gt.flatten(end_dim=1), right.flatten(end_dim=1)

        residue = (left - left_gt).abs().clamp(0, 1)
        grid = make_grid([torch.hstack(column) for column in zip(left, residue, left_gt, right)], nrow=8)

        plt.title("left, abs(left - left_gt), left_gt, right")
        plt.imshow(grid.permute(1, 2, 0))
        plt.xticks([])
        plt.yticks([])
        plt.show()


if __name__ == "__main__":
    datamodule = DataModule("3DMovies", n_frames=2, crop_size=(400, 960), magnitude=0.3, batch_size=3)
    datamodule.plot_example()
