"""Models Training Script

Follow dataset preparation instructions in datasets package.
Make sure that dataset is prepared correctly by running
this command:

```shell
pyton data.py
```

Then use this command to start training:

```shell
python train.py \
    --model=SIMP \  # or DCMC
    --dataset_path=datasets/dataset \
    --accelerator="gpu" \
    --batch_size=16  \
    --img_height=256 \
    --img_width=512  \
    --max_epochs=100  \
    --num_workers=2  \
    --check_val_every_n_epoch=5
```
"""

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
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()

datamodule = CTDataModule(
    image_dir=Path(args.dataset_path),
    batch_size=args.batch_size,
    patch_size=(args.img_height, args.img_width),
    num_workers=args.num_workers)

model = {
    Model.DCMC: DCMC(),
    Model.SIMP: SIMP()
}[args.model]

wandb_logger = WandbLogger(project="color-transfer", log_model="all")

checkpoint = pl.callbacks.ModelCheckpoint(
    monitor="Loss",
    every_n_epochs=25,
    save_top_k=-1)

trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint], logger=wandb_logger)
trainer.fit(model, datamodule)
