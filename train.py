import argparse
from enum import Enum
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from data import CTDataModule
from methods.dcmc import DCMC
from methods.simp import SIMP


class Model(Enum):
    DCMC = "DCMC"
    SIMP = "SIMP"

    def __str__(self):
        return self.value


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=Model, choices=list(Model))
parser.add_argument("--dataset_path", type=str)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--img_height", type=int)
parser.add_argument("--img_width", type=int)
parser.add_argument("--num_workers", type=int)
parser.add_argument("--use_real_distortions", action="store_true")
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()

datamodule = CTDataModule(
    image_dir=Path(args.dataset_path),
    batch_size=args.batch_size,
    patch_size=(args.img_height, args.img_width),
    num_workers=args.num_workers,
    use_real_distortions=args.use_real_distortions)

model = {
    Model.DCMC: DCMC(),
    Model.SIMP: SIMP()
}[args.model]

wandb_logger = WandbLogger(project="color-transfer", log_model=True)
checkpoint = pl.callbacks.ModelCheckpoint(monitor="PSNR", mode="max")

trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint], logger=wandb_logger)
trainer.fit(model, datamodule)
