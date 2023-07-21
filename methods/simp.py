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
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.encoders import get_encoder

from methods.modules import MultiScaleFeatureExtration, CasPAM, MultiScaleTransfer
from methods.modules import warp


class SIMP(pl.LightningModule):
    def __init__(self,
                 encoder_name="efficientnet-b2",
                 encoder_weights=None,
                 matcher_layers=(4, 4, 4, 4, 4, 4),
                 decoder_channels=(512, 256, 128, 64, 32),
                 num_logged_images=3):
        super().__init__()

        self.num_logged_images = num_logged_images

        self.encoder = get_encoder(
            name=encoder_name,
            weights=encoder_weights,
        )

        self.matcher = CasPAM(
            layers=matcher_layers,
            channels=self.encoder.out_channels,
            n_iterpolations_at_end=6 - len(matcher_layers),
        )

        self.decoder = UnetDecoder(
            encoder_channels=[2 * x + 1 for x in self.encoder.out_channels],
            decoder_channels=decoder_channels,
            n_blocks=5,
            use_batchnorm=False,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=3,
            activation=None,
            kernel_size=3,
        )

    def forward(self, left, right):
        features_left = self.encoder(left)
        features_right = self.encoder(right)

        atts, valid_masks = self.matcher(features_left, features_right)

        features = [
            torch.cat([
                feature_left,
                warp(feature_right, att[0]),
                valid_mask[0],
            ], dim=1)
            for feature_left, feature_right, att, valid_mask in
            zip(features_left, features_right, atts, valid_masks)
        ]

        decoder_output = self.decoder(*features)

        return self.segmentation_head(decoder_output), atts[0][0]

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
