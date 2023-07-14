"""Stereo Investigation Model Precise

We based our method on that of Croci et al., borrowing ideas from Wang et al.
Our contribution is an effective multiscale network structure that works
2.6 times faster than Croci’s neural-network-based method and, for artificial
distortions, outperforms it by 3.7 dB on PSNR and, for real-world distortions,
do so by 1.3 dB. Our method consists of three main modules: feature extraction,
cascaded parallax attention, and transfer.

For each scale we kept the channel count unchanged, as this table shows:
## scale     #  1  #  1/2  #  1/4  #  1/8  #  1/16  #  1/32  ##
## channels  #  16 #  32   #  64   #  128   #  256   #  512   ##

Citation
--------
@misc{chistov2023color,
  author={Chistov, Egor and Alutis, Nikita and Velikanov, Maxim and Vatolin, Dmitriy},
  title={Color Mismatches in Stereoscopic Video: Real-World Dataset and Deep Correction Method},
  howpublished={arXiv:2303.06657 [cs.CV]},
  year={2023}
}
"""

import torch
import pytorch_lightning as pl
from kornia.losses import ssim_loss
import torch.nn.functional as F
from piq import psnr, ssim, fsim

from methods.modules import MultiScaleFeatureExtration, CasPAM, MultiScaleTransfer
from methods.modules import cas_outputs, warp


class SIMP(pl.LightningModule):
    def __init__(self,
                 layers=(2, 2, 2, 2),
                 pam_layers=(4, 4, 4, 4),
                 channels=(16, 32, 64, 128, 256, 512),
                 num_logged_images=3):
        super().__init__()

        self.num_logged_images = num_logged_images

        self.extraction = MultiScaleFeatureExtration(layers, channels)
        self.cas_pam = CasPAM(pam_layers, channels[2:])
        self.transfer = MultiScaleTransfer((2, 2) + tuple(layers), channels)

    def forward(self, left, right):
        b, _, h, w = left.shape

        fea_left = self.extraction(left)
        fea_right = self.extraction(right)

        costs = self.cas_pam(fea_left[-4:], fea_right[-4:])

        atts, valid_masks = cas_outputs(costs, n_iterpolations_at_end=2)

        fea_warped_right = [
            warp(image, att[0])
            for image, att in zip(fea_right[:-3], atts[::-1])
        ]

        valid_masks = [x[0] for x in valid_masks[::-1]]
        corrected_left = self.transfer(fea_left[:-3], fea_warped_right, valid_masks)

        return corrected_left, atts[-1][0]

    def training_step(self, batch, batch_idx):
        left, left_gt, right = batch

        corrected_left, _ = self(left, right)

        loss_mse = F.mse_loss(corrected_left, left_gt)
        loss_ssim = ssim_loss(corrected_left, left_gt, window_size=11)

        self.log("MSE Loss", loss_mse)
        self.log("SSIM Loss", loss_ssim)

        return loss_mse + loss_ssim

    def validation_step(self, batch, batch_idx):
        left, left_gt, right = batch

        corrected_left, att = self(left, right)
        corrected_left = corrected_left.clamp(0, 1)
        warped_right = warp(right, att)

        self.log("PSNR", psnr(corrected_left, left_gt))
        self.log("SSIM", ssim(corrected_left, left_gt))  # noqa
        self.log("FSIM", fsim(corrected_left, left_gt))

        if batch_idx == 0 and hasattr(self.logger, "log_image"):
            self.logger.log_image(
                key="Validation",
                images=[batch[:self.num_logged_images]
                        for batch in [left, warped_right, corrected_left, left_gt, right]],
                caption=["Left Distorted", "Warped Right", "Left Corrected", "Left", "Right"])

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.estimated_stepping_batches,
            eta_min=1e-6)

        lr_scheduler_dict = {"scheduler": scheduler, "interval": "step"}

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}
