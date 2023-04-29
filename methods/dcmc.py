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
from kornia.metrics import psnr, ssim
from kornia.losses import ssim_loss
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger

from methods.losses import loss_pam_smoothness, loss_pam_photometric, loss_pam_cycle
from methods.modules import FeatureExtration, PAB, output, Transfer


class DCMC(pl.LightningModule):
    def __init__(self, num_logged_images: int = 4):
        super().__init__()

        self.num_logged_images = num_logged_images

        self.extraction = FeatureExtration()
        self.pam = PAB(channels=64, weighted_shortcut=False)
        self.value = torch.nn.Conv2d(64, 64, kernel_size=1)
        self.transfer = Transfer()

    def forward(self, left, right):
        fea_left = self.extraction(left)
        fea_right = self.extraction(right)

        b, _, h, w = fea_left.shape

        _, _, cost = self.pam(fea_left, fea_right, cost=(0, 0))

        att, att_cycle, valid_mask = output(cost)

        fea_warped_right = torch.matmul(att[0], self.value(fea_right).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        corrected_left = self.transfer(fea_left, fea_warped_right, valid_mask)

        warped_right = torch.matmul(att[0], right.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return corrected_left, (
            att,
            att_cycle,
            valid_mask,
            warped_right
        )

    def training_step(self, batch, batch_idx):
        left, left_gt, right = batch

        corrected_left, (att, att_cycle, valid_mask, _) = self(left, right)

        loss_pm = loss_pam_photometric(left, right, att, valid_mask)
        loss_smooth = 0.1 * loss_pam_smoothness(att)
        loss_cycle = loss_pam_cycle(att_cycle, valid_mask)

        loss_cc = F.l1_loss(corrected_left, left_gt) + \
            F.mse_loss(corrected_left, left_gt) + \
            ssim_loss(corrected_left, left_gt, window_size=11)

        loss = loss_cc + 0.005 * (loss_pm + loss_smooth + loss_cycle)

        self.log("Photometric Loss", 0.005 * loss_pm)
        self.log("Smoothness Loss",  0.005 * loss_smooth)
        self.log("Cycle Loss",  0.005 * loss_cycle)
        self.log("Color Correction Loss", loss_cc)
        self.log("Loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        left, left_gt, right = batch

        corrected_left, (_, _, _, warped_right) = self(left, right)

        psnr_value = psnr(corrected_left, left_gt, max_val=1)
        ssim_value = ssim(corrected_left, left_gt, window_size=11).mean()

        self.log("PSNR", psnr_value)
        self.log("SSIM", ssim_value)

        if batch_idx == 0 and isinstance(self.logger, WandbLogger):
            self.logger.log_image(
                key="Validation",
                images=[batch[:self.num_logged_images]
                        for batch in [left, warped_right, corrected_left.clamp(0, 1), left_gt, right]],
                caption=["Left Distorted", "Warped Right", "Left Corrected", "Left", "Right"])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        return [optimizer]
