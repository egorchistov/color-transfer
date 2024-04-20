"""Deep Color Mismatch Correction in Stereoscopic 3D Images

Citation
--------
@inproceedings{croci2021deep,
  title={Deep Color Mismatch Correction In Stereoscopic 3d Images},
  author={Croci, Simone and Ozcinar, Cagri and Zerman, Emin and Dudek, Roman and Knorr, Sebastian and Smolic, Aljosa},
  booktitle={2021 IEEE International Conference on Image Processing (ICIP)},
  pages={1749--1753},
  year={2021},
  organization={IEEE}
}
"""

import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from piq import psnr, ssim, fsim
from kornia.losses import ssim_loss

from pasmnet.attention import PAB
from pasmnet.backbone import ResB
from pasmnet.losses import loss_pam_photometric, loss_pam_cycle, loss_pam_smoothness
from pasmnet.utils import warp, output, regress_disp
from utils.icid import icid
from utils.visualizations import chess_mix, rgbmse, rgbssim


class DCMCS3DI(pl.LightningModule):
    def __init__(self,
                 extraction_layers=18,
                 transfer_layers=6,
                 channels=64,
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.max_scores = {}

        channels = self.hparams.channels

        self.extraction = torch.nn.Sequential(torch.nn.Conv2d(3, channels, kernel_size=3, padding=1))
        for _ in range(self.hparams.extraction_layers):
            self.extraction.append(ResB(channels, channels))

        self.matcher = PAB(channels)

        self.transfer = torch.nn.Sequential(torch.nn.Conv2d(2 * channels + 1, channels, kernel_size=1))
        for _ in range(self.hparams.transfer_layers):
            self.transfer.append(ResB(channels, channels))
        self.transfer.append(torch.nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1))
        self.transfer.append(torch.nn.Conv2d(channels // 2, 3, kernel_size=3, padding=1))

    def forward(self, left, right):
        fea_left = self.extraction(left)
        fea_right = self.extraction(right)

        att, att_cycle, valid_mask = output(self.matcher(fea_left, fea_right))
        fea_warped_right = warp(self.matcher.value(fea_right), att[0])
        corrected_left = self.transfer(torch.cat([fea_left, fea_warped_right, valid_mask[0]], dim=1))

        return corrected_left, (
            att,
            att_cycle,
            valid_mask,
            warp(right, att[0])
        )

    def step(self, batch, prefix):
        corrected_left, (att, att_cycle, valid_mask, _) = self(batch["target"], batch["reference"])

        loss_l1 = F.l1_loss(corrected_left, batch["gt"])
        loss_mse = F.mse_loss(corrected_left, batch["gt"])
        loss_ssim = ssim_loss(corrected_left, batch["gt"], window_size=11)

        loss_pm = 0.005 * loss_pam_photometric(batch["target"], batch["reference"], att, valid_mask)
        loss_cycle = 0.005 * loss_pam_cycle(att_cycle, valid_mask)
        loss_smooth = 0.005 * loss_pam_smoothness(att)

        self.log(f"{prefix} L1 Loss", loss_l1)
        self.log(f"{prefix} MSE Loss", loss_mse)
        self.log(f"{prefix} SSIM Loss", loss_ssim)

        self.log(f"{prefix} Photometric Loss", loss_pm)
        self.log(f"{prefix} Cycle Loss",  loss_cycle)
        self.log(f"{prefix} Smoothness Loss",  loss_smooth)

        corrected_left = corrected_left.clamp(0, 1)

        self.log(f"{prefix} PSNR", psnr(corrected_left, batch["gt"]))
        self.log(f"{prefix} SSIM", ssim(corrected_left, batch["gt"]))  # noqa
        self.log(f"{prefix} FSIM", fsim(corrected_left, batch["gt"]))
        self.log(f"{prefix} iCID", icid(corrected_left, batch["gt"]))

        return loss_l1 + loss_mse + loss_ssim + loss_pm + loss_cycle + loss_smooth

    def training_step(self, batch, batch_idx):
        return self.step(batch, prefix="Training")

    def validation_step(self, batch, batch_idx):
        self.step(batch, prefix="Validation")

    def test_step(self, batch, batch_idx):
        self.step(batch, prefix="Test")

    def on_train_epoch_end(self):
        super().on_train_epoch_end()

        batch = next(iter(self.trainer.train_dataloader))
        self.log_images(batch, prefix="Training")

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()

        batch = next(iter(self.trainer.val_dataloaders))
        self.log_images(batch, prefix="Validation")

    def log_images(self, batch, prefix):
        if (hasattr(self.logger, "log_image") and
                self.trainer.logged_metrics[f"{prefix} PSNR"] > self.max_scores.get(prefix, 0)):
            self.max_scores[prefix] = self.trainer.logged_metrics[f"{prefix} PSNR"]

            batch = {k: v[-1].unsqueeze(dim=0).to(self.device) for k, v in batch.items()}

            result, (att, _, valid_mask, warped_right) = self(batch["target"], batch["reference"])
            result = result.clamp(0, 1)

            disparity = regress_disp(att[0], valid_mask[0].float())
            occlusion_mask = (1 - valid_mask[0].float()).squeeze().cpu().numpy() * 255

            data = {
                "Left Ground Truth/Corrected": chess_mix(batch["gt"], result),
                "RGB MSE Error": rgbmse(batch["gt"], result),
                "RGB SSIM Error": rgbssim(batch["gt"], result),
                "Disparity": disparity,
                "Warped Right": warped_right,
            }

            mask = {"Occlusions": {"mask_data": occlusion_mask, "class_labels": {255: "Occlusions"}}}

            self.logger.log_image(
                key=f"{prefix} Images",
                images=list(data.values()),
                caption=list(data.keys()),
                masks=[None] * (len(data) - 1) + [mask]
            )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        return {"optimizer": optimizer}
