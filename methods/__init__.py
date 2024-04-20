import importlib

import torch
import pytorch_lightning as pl
from piq import psnr, ssim, fsim

from utils.icid import icid


class Runner(pl.LightningModule):
    def __init__(self, func_spec):
        super().__init__()

        specs = func_spec.split(".")
        module, func = ".".join(specs[:-1]), specs[-1]
        self.func = importlib.import_module(module).__getattribute__(func)

    def forward(self, batch):
        outputs = []
        for target, reference in zip(batch["target"], batch["reference"]):
            target = target.permute(1, 2, 0).detach().cpu().numpy()
            reference = reference.permute(1, 2, 0).detach().cpu().numpy()

            output = torch.from_numpy(self.func(target, reference)).float().permute(2, 0, 1)
            outputs.append(output)

        return torch.stack(outputs)

    def test_step(self, batch, batch_idx):
        result = self(batch).clamp(0, 1)

        psnr_value = psnr(result, batch["gt"])
        ssim_value = ssim(result, batch["gt"])
        fsim_value = fsim(result, batch["gt"])
        icid_value = icid(result, batch["gt"])

        self.log("Test PSNR", psnr_value, prog_bar=True)
        self.log("Test SSIM", ssim_value)  # noqa
        self.log("Test FSIM", fsim_value)
        self.log("Test iCID", icid_value)
