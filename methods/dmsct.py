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


class DMSCT(pl.LightningModule):
    def __init__(self,
                 encoder_name="efficientnet-b2",
                 encoder_depth=4,
                 encoder_weights=None,
                 decoder_channels=(256, 128, 64, 32),
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.max_scores = {}

        self.gmflow = GMFlow(pretrained="mixdata")
        for p in self.gmflow.parameters():
            p.requires_grad = False

        self.encoder = get_encoder(
            name=self.hparams.encoder_name,
            depth=self.hparams.encoder_depth,
            weights=self.hparams.encoder_weights,
        )

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

        self.head = SegmentationHead(
            in_channels=self.hparams.decoder_channels[-1],
            out_channels=3,
        )

    @staticmethod
    def derive_matcher_inference_size(shape, max_area=500 * 900, padding_factor=32):
        inference_size = [int(np.ceil(shape[-2] / padding_factor)) * padding_factor,
                          int(np.ceil(shape[-1] / padding_factor)) * padding_factor]

        aspect_ratio = shape[-1] / shape[-2]

        max_h = np.floor(np.sqrt(max_area / aspect_ratio))
        max_w = np.floor(max_h * aspect_ratio)

        max_inference_size = [int(np.ceil(max_h / padding_factor)) * padding_factor,
                              int(np.ceil(max_w / padding_factor)) * padding_factor]

        if inference_size[0] * inference_size[1] > max_inference_size[0] * max_inference_size[1]:
            inference_size = max_inference_size

        return inference_size

    def derive_pad_size(self, shape):
        padding_factor = 2 ** self.hparams.encoder_depth

        pad_size = [0, (shape[-2] % padding_factor != 0) * (padding_factor - shape[-2] % padding_factor),
                    0, (shape[-1] % padding_factor != 0) * (padding_factor - shape[-1] % padding_factor)]

        return pad_size

    def forward(self, target, reference):
        matcher_inference_size = DMSCT.derive_matcher_inference_size(reference.shape)

        with torch.no_grad():
            matcher_dict = self.gmflow(
                target * 255,
                reference * 255,
                inference_size=matcher_inference_size,
                pred_bidir_flow=True,
                fwd_bwd_consistency_check=True,
            )

        _, _, height, width = reference.shape
        pad_size = self.derive_pad_size(reference.shape)

        matcher_dict["flow"] = torch.nn.functional.pad(matcher_dict["flow"], pad_size)
        matcher_dict["fwd_occ"] = torch.nn.functional.pad(matcher_dict["fwd_occ"], pad_size)
        features_target = self.encoder(torch.nn.functional.pad(target, pad_size))
        features_reference = self.encoder(torch.nn.functional.pad(reference, pad_size))

        print(
            matcher_dict["flow"].shape,
            matcher_dict["fwd_occ"].shape,
            features_target[0].shape,
            features_reference[0].shape,
        )

        features = [
            torch.cat([
                feature_target,
                flow_warp(feature_reference, self.gmflow.upsample_flow(matcher_dict["flow"], feature=None, bilinear=True, upsample_factor=2 ** -idx)),
                torch.nn.functional.interpolate((1 - matcher_dict["fwd_occ"]), mode="nearest", scale_factor=2 ** -idx)
            ], dim=1)
            for idx, (feature_target, feature_reference) in enumerate(zip(
                features_target,
                features_reference
            ))
        ]

        return target + self.head(self.decoder(*features))[:, :, :height, :width]

    def step(self, batch, prefix):
        result = self(batch["target"], batch["reference"])

        loss_mse = mse_loss(result, batch["gt"])
        loss_ssim = 0.1 * ssim_loss(result, batch["gt"], window_size=11)

        self.log(f"{prefix} MSE Loss", loss_mse)
        self.log(f"{prefix} SSIM Loss", loss_ssim)
        self.log(f"{prefix} PSNR", psnr(result.clamp(0, 1), batch["gt"]), prog_bar=True)
        self.log(f"{prefix} SSIM", ssim(result.clamp(0, 1), batch["gt"]))  # noqa

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
                self.trainer.logged_metrics[f"{prefix} PSNR"] > self.max_scores.get(prefix, 0)):
            self.max_scores[prefix] = self.trainer.logged_metrics[f"{prefix} PSNR"]

            batch = {k: v[-1].unsqueeze(dim=0).to(self.device) for k, v in batch.items()}

            matcher_dict = self.gmflow(
                batch["target"] * 255,
                batch["reference"] * 255,
                pred_bidir_flow=True,
                pred_flow_viz=True,
                fwd_bwd_consistency_check=True,
            )

            flow_viz = torch.from_numpy(matcher_dict["flow_viz"]) / 255
            warped_right = flow_warp(batch["reference"], matcher_dict["flow"])
            occlusion_mask = matcher_dict["fwd_occ"].squeeze().cpu().numpy() * 255

            result = self(batch["target"], batch["reference"]).clamp(0, 1)

            data = {
                "Left Ground Truth/Corrected": chess_mix(batch["gt"], result),
                "RGB MSE Error": rgbmse(batch["gt"], result),
                "RGB SSIM Error": rgbssim(batch["gt"], result),
                "Optical Flow": flow_viz,
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.estimated_stepping_batches,
            eta_min=1e-6)

        lr_scheduler_dict = {"scheduler": scheduler, "interval": "step"}

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}
