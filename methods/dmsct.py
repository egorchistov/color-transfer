import numpy as np
import torch
import pytorch_lightning as pl
from kornia.losses import ssim_loss
from torch.nn.functional import mse_loss
from piq import psnr, ssim
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.encoders import get_encoder

from utils.visualizations import chess_mix, rgbmse, rgbssim
from unimatch import GMFlow
from unimatch.geometry import flow_warp


class ConvGRU(torch.nn.Module):
    """https://github.com/PeterL1n/RobustVideoMatting/blob/master/model/decoder.py"""
    def __init__(self,
                 channels: int,
                 kernel_size: int = 3,
                 padding: int = 1):
        super().__init__()
        self.channels = channels
        self.ih = torch.nn.Sequential(
            torch.nn.Conv2d(channels * 2, channels * 2, kernel_size, padding=padding),
            torch.nn.Sigmoid()
        )
        self.hh = torch.nn.Sequential(
            torch.nn.Conv2d(channels * 2, channels, kernel_size, padding=padding),
            torch.nn.Tanh()
        )

    def forward_single_frame(self, x, h):
        r, z = self.ih(torch.cat([x, h], dim=1)).split(self.channels, dim=1)
        c = self.hh(torch.cat([x, r * h], dim=1))
        h = (1 - z) * h + z * c
        return h, h

    def forward_time_series(self, x, h):
        o = []
        for xt in x.unbind(dim=1):
            ot, h = self.forward_single_frame(xt, h)
            o.append(ot)
        o = torch.stack(o, dim=1)
        return o, h

    def forward(self, x, h=None):
        if h is None:
            h = torch.zeros((x.size(0), x.size(-3), x.size(-2), x.size(-1)),
                            device=x.device, dtype=x.dtype)

        if x.ndim == 5:
            return self.forward_time_series(x, h)
        else:
            return self.forward_single_frame(x, h)


class DMSCT(pl.LightningModule):
    def __init__(self,
                 encoder_name="efficientnet-b2",
                 encoder_depth=4,
                 encoder_weights=None,
                 decoder_channels=(256, 128, 64, 32),
                 use_gru=False,
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.max_psnrs = {
            "Training": 0,
            "Validation": 0,
        }

        self.encoder = get_encoder(
            name=self.hparams.encoder_name,
            depth=self.hparams.encoder_depth,
            weights=self.hparams.encoder_weights,
        )

        self.gmflow = GMFlow(pretrained="mixdata")
        for p in self.gmflow.parameters():
            p.requires_grad = False

        encoder_out_channels = list(self.encoder.out_channels)
        encoder_out_channels = [
            2 * channels + 1
            for channels in encoder_out_channels
        ]

        self.decoder = UnetDecoder(
            encoder_channels=encoder_out_channels,
            decoder_channels=self.hparams.decoder_channels,
            n_blocks=self.hparams.encoder_depth,
            use_batchnorm=False,
        )

        if self.hparams.use_gru:
            self.gru = ConvGRU(
                channels=self.hparams.decoder_channels[-1],
            )

        self.head = SegmentationHead(
            in_channels=self.hparams.decoder_channels[-1],
            out_channels=3,
        )

    def forward(self, left, right, h=None):
        is_video = left.ndim == 5

        if is_video:
            B, T, _, _, _ = left.shape

            left = left.flatten(end_dim=1)
            right = right.flatten(end_dim=1)

        concat = torch.cat((left, right), dim=0)  # [2BT, C, H, W]
        features = self.encoder(concat)  # list of [2BT, C, H, W], resolution from high to low

        feature0, feature1 = [], []

        for i in range(len(features)):
            feature = features[i]
            chunks = torch.chunk(feature, 2, 0)  # tuple
            feature0.append(chunks[0])
            feature1.append(chunks[1])

        features_left = feature0
        features_right = feature1

        padding_factor = 32
        inference_size = [int(np.ceil(left.shape[-2] / padding_factor)) * padding_factor,
                          int(np.ceil(left.shape[-1] / padding_factor)) * padding_factor]

        aspect_ratio = left.shape[-1] / left.shape[-2]

        max_h = np.floor(np.sqrt(500 * 900 / aspect_ratio))
        max_w = np.floor(max_h * aspect_ratio)

        max_inference_size = [int(np.ceil(max_h / padding_factor)) * padding_factor,
                              int(np.ceil(max_w / padding_factor)) * padding_factor]

        if inference_size[0] * inference_size[1] > max_inference_size[0] * max_inference_size[1]:
            inference_size = max_inference_size

        out = self.gmflow(left * 255,
                          right * 255,
                          inference_size=inference_size,
                          pred_bidir_flow=True,
                          fwd_bwd_consistency_check=True,
                          )

        features = [
            torch.cat([
                feature_left,
                flow_warp(feature_right, self.gmflow.upsample_flow(out["flow"], feature=None, bilinear=True, upsample_factor=2 ** -idx)),
                torch.nn.functional.interpolate((1 - out["fwd_occ"]), mode="nearest", scale_factor=2 ** -idx)
            ], dim=1)
            for idx, (feature_left, feature_right) in enumerate(zip(
                features_left,
                features_right
            ))
        ]

        decoder_output = self.decoder(*features)

        if is_video and self.hparams.use_gru:
            decoder_output = decoder_output.unflatten(dim=0, sizes=(B, T))
            decoder_output, h = self.gru(decoder_output, h)
            decoder_output = decoder_output.flatten(end_dim=1)

        cleft = self.head(decoder_output)

        if is_video:
            cleft = cleft.unflatten(dim=0, sizes=(B, T))

        return left + cleft, h

    def step(self, batch, prefix):
        left, left_gt, right = batch

        corrected_left, _ = self(left, right)

        if left.ndim == 5:
            left_gt = left_gt.flatten(end_dim=1)
            corrected_left = corrected_left.flatten(end_dim=1)

        loss_mse = mse_loss(corrected_left, left_gt)
        loss_ssim = 0.1 * ssim_loss(corrected_left, left_gt, window_size=11)

        self.log(f"{prefix} MSE Loss", loss_mse)
        self.log(f"{prefix} SSIM Loss", loss_ssim)
        self.log(f"{prefix} PSNR", psnr(corrected_left.clamp(0, 1), left_gt), prog_bar=prefix == "Training")
        self.log(f"{prefix} SSIM", ssim(corrected_left.clamp(0, 1), left_gt))  # noqa

        return loss_mse + loss_ssim

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

    def on_test_epoch_end(self):
        super().on_test_epoch_end()

        batch = next(iter(self.trainer.test_dataloaders))

        self.log_images(batch, prefix="Test")

    def log_images(self, batch, prefix):
        if (hasattr(self.logger, "log_image") and
                self.trainer.logged_metrics[f"{prefix} PSNR"] > self.max_psnrs[prefix]):
            self.max_psnrs[prefix] = self.trainer.logged_metrics[f"{prefix} PSNR"]

            left, left_gt, right = (view[0].unsqueeze(dim=0).to(self.device) for view in batch)

            if left.ndim == 5:
                left, left_gt, right = (view[:, 0] for view in (left, left_gt, right))

            corrected_left, _ = self(left, right)
            corrected_left = corrected_left.clamp(0, 1)

            out = self.gmflow(left * 255, right * 255,
                              pred_bidir_flow=True,
                              pred_flow_viz=True,
                              fwd_bwd_consistency_check=True,
                              )

            flow_viz = torch.from_numpy(out["flow_viz"]) / 255
            warped_right = flow_warp(right, out["flow"])

            data = {
                "Left Ground Truth/Corrected": chess_mix(left_gt, corrected_left),
                "RGB MSE Error": rgbmse(left_gt, corrected_left),
                "RGB SSIM Error": rgbssim(left_gt, corrected_left),
                "Optical Flow": flow_viz,
            }

            self.logger.log_image(key=f"{prefix} Images", images=list(data.values()), caption=list(data.keys()))
            self.logger.log_image(key=f"{prefix} Images", images=[warped_right], caption=["Warped Right"],
                                  masks={"Occlusions": {"mask_data": out["fwd_occ"]}})

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.estimated_stepping_batches,
            eta_min=1e-6)

        lr_scheduler_dict = {"scheduler": scheduler, "interval": "step"}

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}
