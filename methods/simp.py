"""Stereo Investigation Model Precise

We based our method on that of Croci et al., borrowing ideas from Wang et al.
Our contribution is an effective multiscale network structure that works
2.6 times faster than Croci’s neural-network-based method and, for artificial
distortions, outperforms it by 3.7 dB on PSNR and, for real-world distortions,
do so by 1.3 dB. Our method consists of three main modules: feature extraction,
cascaded parallax attention, and transfer.

For each scale we kept the channel count unchanged, as this table shows:
## scale     #  1  #  1/2  #  1/4  #  1/8  #  1/16  #  1/32  ##
## channels  #  16 #  32   #  64   #  96   #  128   #  160   ##

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
from kornia.metrics import psnr, ssim
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger

from methods.losses import loss_pam_photometric_multiscale, loss_pam_cycle_multiscale, loss_pam_smoothness_multiscale
from methods.modules import MultiScaleFeatureExtration, CasPAM, output, MultiScaleTransfer


class SIMP(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.extraction = MultiScaleFeatureExtration()
        self.cas_pam = CasPAM()
        self.transfer = MultiScaleTransfer()

    def forward(self, left, right):
        b, _, h, w = left.shape

        fea_left = self.extraction(left)
        fea_right = self.extraction(right)

        costs = self.cas_pam(fea_left[-3:], fea_right[-3:])

        att_s1, att_cycle_s1, valid_mask_s1 = output(costs[0])
        att_s2, att_cycle_s2, valid_mask_s2 = output(costs[1])
        att_s3, att_cycle_s3, valid_mask_s3 = output(costs[2])

        # PAM_stage at 1/2 and 1 scales consumes too much memory
        att_s4_0 = F.interpolate(att_s3[0].unsqueeze(1), scale_factor=2, mode="trilinear", align_corners=False).squeeze(1)
        att_s5_0 = F.interpolate(att_s3[0].unsqueeze(1), scale_factor=4, mode="trilinear", align_corners=False).squeeze(1)
        att_s0_0 = F.interpolate(att_s1[0].unsqueeze(1), scale_factor=0.5, mode="trilinear", align_corners=False).squeeze(1)

        fea_warped_right = [
            torch.matmul(att, image.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) for image, att in zip(
                fea_right[:-3],
                [att_s5_0, att_s4_0, att_s3[0], att_s2[0], att_s1[0], att_s0_0]
            )
        ]

        valid_masks = [
            F.interpolate(valid_mask_s3[0].float(), scale_factor=4, mode="nearest"),
            F.interpolate(valid_mask_s3[0].float(), scale_factor=2, mode="nearest"),
            valid_mask_s3[0],
            valid_mask_s2[0],
            valid_mask_s1[0],
            F.interpolate(valid_mask_s1[0].float(), scale_factor=0.5, mode="nearest")
        ]

        corrected_left = self.transfer(fea_left[:-3], fea_warped_right, valid_masks)

        warped_right = torch.matmul(att_s5_0, right.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return corrected_left, (
            [att_s1, att_s2, att_s3],
            [att_cycle_s1, att_cycle_s2, att_cycle_s3],
            [valid_mask_s1, valid_mask_s2, valid_mask_s3],
            warped_right
        )

    def training_step(self, batch, batch_idx):
        left, left_gt, right = batch

        corrected_left, (att, att_cycle, valid_mask, _) = self(left, right)

        loss_pm = loss_pam_photometric_multiscale(left, right, att, valid_mask)
        loss_smooth = 0.1 * loss_pam_smoothness_multiscale(att)
        loss_cycle = loss_pam_cycle_multiscale(att_cycle, valid_mask)

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
                images=[left, warped_right, corrected_left.clamp(0, 1), left_gt, right],
                caption=["Left Distorted", "Warped Right", "Left Corrected", "Left", "Right"])

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.estimated_stepping_batches,
            eta_min=1e-6)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"}
        }
