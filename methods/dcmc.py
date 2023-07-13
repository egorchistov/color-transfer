"""Deep Color Mismatch Correction in Stereoscopic 3D Images

Croci et al. employed a convolutional neural network for color-mismatch
correction. First, the network extracts features from the input stereopair.
It then feeds the extracted features into the parallax-attention mechanism,
which performs stereo matching. Matched features pass through six residual
blocks to yield the corrected stereoscopic view.

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
import pytorch_lightning as pl
from piq import psnr, ssim, fsim
from kornia.losses import ssim_loss
import torch.nn.functional as F

from methods.losses import loss_pam_smoothness, loss_pam_photometric, loss_pam_cycle
from methods.modules import FeatureExtration, PAB, Transfer
from methods.modules import output, warp


class DCMC(pl.LightningModule):
    def __init__(self,
                 extraction_layers=18,
                 transfer_layers=6,
                 channels=64,
                 num_logged_images=3):
        super().__init__()

        self.num_logged_images = num_logged_images

        self.extraction = FeatureExtration(layers=extraction_layers, channels=channels)
        self.pam = PAB(channels=channels, weighted_shortcut=False)
        self.value = torch.nn.Conv2d(channels, channels, kernel_size=1)
        self.transfer = Transfer(layers=transfer_layers, channels=channels)

    def forward(self, left, right):
        fea_left = self.extraction(left)
        fea_right = self.extraction(right)

        b, _, h, w = fea_left.shape

        _, _, cost = self.pam(fea_left, fea_right, cost=(0, 0))

        att, att_cycle, valid_mask = output(cost)

        fea_warped_right = warp(fea_right, att[0])

        corrected_left = self.transfer(fea_left, fea_warped_right, valid_mask)

        warped_right = warp(right, att[0])

        return corrected_left, (
            att,
            att_cycle,
            valid_mask,
            warped_right
        )

    def training_step(self, batch, batch_idx):
        left, left_gt, right = batch

        corrected_left, (att, att_cycle, valid_mask, _) = self(left, right)

        loss_l1 = F.l1_loss(corrected_left, left_gt)
        loss_mse = F.mse_loss(corrected_left, left_gt)
        loss_ssim = ssim_loss(corrected_left, left_gt, window_size=11)

        loss_pm = 0.005 * loss_pam_photometric(left, right, att, valid_mask)
        loss_cycle = 0.005 * loss_pam_cycle(att_cycle, valid_mask)
        loss_smooth = 0.0005 * loss_pam_smoothness(att)

        loss = loss_l1 + loss_mse + loss_ssim + loss_pm + loss_cycle + loss_smooth

        self.log("L1 Loss", loss_l1)
        self.log("MSE Loss", loss_mse)
        self.log("SSIM Loss", loss_ssim)

        self.log("Photometric Loss", loss_pm)
        self.log("Cycle Loss",  loss_cycle)
        self.log("Smoothness Loss",  loss_smooth)

        self.log("Loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        left, left_gt, right = batch

        corrected_left, (_, _, _, warped_right) = self(left, right)
        corrected_left = corrected_left.clamp(0, 1)

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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        return {"optimizer": optimizer}
