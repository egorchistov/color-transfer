import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from methods.dcmc.data import DCMCDataModule
from methods.dcmc import DCMC


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--img_height", type=int)
parser.add_argument("--img_width", type=int)
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()

datamodule = DCMCDataModule(
    Path(args.dataset_path),
    batch_size=args.batch_size,
    patch_size=(args.img_height, args.img_width))

model = DCMC()

wandb_logger = WandbLogger(project="dcmc", log_model="all")
wandb_logger.watch(model, log="all")

checkpoint = pl.callbacks.ModelCheckpoint(
    monitor="Photometric Loss",
    save_last=True)

trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint], logger=wandb_logger)
trainer.fit(model, datamodule)
