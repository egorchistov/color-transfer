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

from methods.gru import ConvGRU
from methods.modules import CasPAM
from methods.modules import warp


class SIMP(pl.LightningModule):
    def __init__(self,
                 encoder_name="efficientnet-b2",
                 encoder_depth=4,
                 encoder_weights=None,
                 matcher_skip_idx=2,
                 matcher_layers=(4, 4, 4),
                 decoder_channels=(256, 128, 64, 32),
                 num_logged_images=3):
        super().__init__()
        self.save_hyperparameters()

        assert self.hparams.matcher_skip_idx + len(self.hparams.matcher_layers) == self.hparams.encoder_depth + 1

        self.encoder = get_encoder(
            name=self.hparams.encoder_name,
            depth=self.hparams.encoder_depth,
            weights=self.hparams.encoder_weights,
        )

        self.matcher = CasPAM(
            layers=self.hparams.matcher_layers,
            channels=self.encoder.out_channels[self.hparams.matcher_skip_idx:]
        )

        encoder_out_channels = list(self.encoder.out_channels)
        encoder_out_channels[self.hparams.matcher_skip_idx:] = [
            2 * channels + 1
            for channels in encoder_out_channels[self.hparams.matcher_skip_idx:]
        ]

        self.decoder = UnetDecoder(
            encoder_channels=encoder_out_channels,
            decoder_channels=self.hparams.decoder_channels,
            n_blocks=self.hparams.encoder_depth,
            use_batchnorm=False,
        )

        self.gru = ConvGRU(
            channels=self.hparams.decoder_channels[-1],
        )

        self.head = SegmentationHead(
            in_channels=self.hparams.decoder_channels[-1],
            out_channels=3,
        )

    def forward(self, left, right, h=None):
        B, T, _, _, _ = left.shape

        left = left.flatten(end_dim=1)
        right = right.flatten(end_dim=1)

        features_left = self.encoder(left)
        features_right = self.encoder(right)

        atts, valid_masks = self.matcher(
            features_left[self.hparams.matcher_skip_idx:],
            features_right[self.hparams.matcher_skip_idx:]
        )

        features_left[self.hparams.matcher_skip_idx:] = [
            torch.cat([
                feature_left,
                warp(feature_right, att[0]),
                valid_mask[0]
            ], dim=1)
            for feature_left, feature_right, att, valid_mask in
            zip(
                features_left[self.hparams.matcher_skip_idx:],
                features_right[self.hparams.matcher_skip_idx:],
                atts,
                valid_masks
            )
        ]

        decoder_output = self.decoder(*features_left)

        decoder_output, h = self.gru(decoder_output.unflatten(dim=0, sizes=(B, T)), h)

        cleft = self.head(decoder_output.flatten(end_dim=1))

        return cleft.unflatten(dim=0, sizes=(B, T)), h

    def training_step(self, batch, batch_idx):
        left, left_gt, right = batch

        corrected_left, _ = self(left, right)

        left_gt = left_gt.flatten(end_dim=1)
        corrected_left = corrected_left.flatten(end_dim=1)

        loss_mse = F.mse_loss(corrected_left, left_gt)
        loss_ssim = ssim_loss(corrected_left, left_gt, window_size=11)

        self.log("MSE Loss", loss_mse)
        self.log("SSIM Loss", loss_ssim)

        return loss_mse + loss_ssim

    def validation_step(self, batch, batch_idx):
        left, left_gt, right = batch

        _, T, _, _, _ = left.shape

        corrected_left, _ = self(left, right)
        corrected_left = corrected_left.clamp(0, 1)

        left = left.flatten(end_dim=1)
        left_gt = left_gt.flatten(end_dim=1)
        right = right.flatten(end_dim=1)
        corrected_left = corrected_left.flatten(end_dim=1)

        self.log("PSNR", psnr(corrected_left, left_gt))
        self.log("SSIM", ssim(corrected_left, left_gt))  # noqa
        self.log("FSIM", fsim(corrected_left, left_gt))

        if batch_idx == 0 and hasattr(self.logger, "log_image"):
            self.logger.log_image(
                key="Validation",
                images=[batch[:self.hparams.num_logged_images * T: T]
                        for batch in [left, corrected_left, left_gt, right]],
                caption=["Left Distorted", "Left Corrected", "Left", "Right"])

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.estimated_stepping_batches,
            eta_min=1e-6)

        lr_scheduler_dict = {"scheduler": scheduler, "interval": "step"}

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}
