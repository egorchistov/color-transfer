import numpy as np
import torch
import pytorch_lightning as pl
from kornia.losses import ssim_loss
from torch.nn.functional import mse_loss
from piq import psnr, ssim
from segmentation_models_pytorch import Unet
from torchvision.transforms import ColorJitter

from unimatch import GMFlow
from unimatch.geometry import flow_warp


class DeepColorTransfer(torch.nn.Module):
    def __init__(self, max_hw=480 * 896):
        super().__init__()
        self.max_hw = max_hw

        self.matcher = GMFlow(pretrained="mixdata")
        for p in self.matcher.parameters():
            p.requires_grad = False

        self.transfer = Unet(in_channels=7, classes=3)

    @staticmethod
    def derive_inference_size(shape, max_hw, padding_factor=32):
        inference_size = [int(np.ceil(shape[-2] / padding_factor)) * padding_factor,
                          int(np.ceil(shape[-1] / padding_factor)) * padding_factor]

        aspect_ratio = shape[-1] / shape[-2]

        max_h = np.floor(np.sqrt(max_hw / aspect_ratio))
        max_w = np.floor(max_h * aspect_ratio)

        max_inference_size = [int(np.ceil(max_h / padding_factor)) * padding_factor,
                              int(np.ceil(max_w / padding_factor)) * padding_factor]

        if inference_size[0] * inference_size[1] > max_inference_size[0] * max_inference_size[1]:
            inference_size = max_inference_size

        return inference_size

    @torch.no_grad()
    def match_reference(self, batch, use_gt=False):
        inference_size = DeepColorTransfer.derive_inference_size(batch["reference"].shape, self.max_hw)

        matcher_dict = self.matcher(
            batch["gt"] * 255 if use_gt else batch["target"] * 255,
            batch["reference"] * 255,
            inference_size=inference_size,
            pred_bidir_flow=True,
            fwd_bwd_consistency_check=True,
        )

        batch["matched_reference"] = flow_warp(batch["reference"], matcher_dict["flow"])
        batch["valid_mask"] = 1 - matcher_dict["fwd_occ"]
        del batch["reference"]

        return batch

    def run_transfer(self, batch):
        features = torch.cat([
            batch["target"],
            batch["matched_reference"],
            batch["valid_mask"],
        ], dim=1)

        return batch["target"] + self.transfer(features)

    def forward(self, batch):
        batch = self.match_reference(batch)
        result = self.run_transfer(batch)

        return result


class TrainDeepColorTransfer(pl.LightningModule):
    def __init__(self, n_patches, patch_shape, *args, **kwargs):
        super().__init__()
        self.n_patches = n_patches
        self.patch_shape = patch_shape

        self.model = DeepColorTransfer(*args, **kwargs)
        self.distort = ColorJitter(0.5, 0.5, 0.5, 0.1)

    @staticmethod
    def view_as_blocks(tensor, block_shape):
        return (tensor
                .unfold(2, block_shape[0], block_shape[0])
                .unfold(3, block_shape[1], block_shape[1])
                .reshape(-1, tensor.shape[1], *block_shape))

    def create_patches(self, batch):
        _, _, height, width = batch["gt"].shape

        # Crop the rest of the frames
        height -= height % self.patch_shape[0]
        width -= width % self.patch_shape[1]

        # Cut the frames into patches
        batch = {
            k: TrainDeepColorTransfer.view_as_blocks(frame[:, :, :height, :width], self.patch_shape)
            for k, frame in batch.items()
        }

        return batch

    def training_step(self, batch, batch_idx):
        # Create patches for `gt`, `matched_reference`, `valid_mask`
        batch = self.model.match_reference(batch, use_gt=True)
        batch = self.create_patches(batch)

        # Select `batch_size` patches with the most confidence
        indexes = torch.argsort(batch["valid_mask"].mean(dim=(-1, -2, -3)), descending=True)
        indexes = indexes[:min(self.n_patches, batch["gt"].shape[0])]
        batch = {k: frame[indexes] for k, frame in batch.items()}

        # Apply uniformly sampled distortions
        batch["target"] = self.distort(batch["gt"])

        # Run Deep Color Transfer
        result = self.model.run_transfer(batch)

        # Calculate and log losses
        loss_mse = mse_loss(result, batch["gt"])
        loss_ssim = 0.1 * ssim_loss(result, batch["gt"], window_size=11)
        self.log("Training MSE Loss", loss_mse)
        self.log("Training SSIM Loss", loss_ssim)

        # Calculate and log metrics
        result = result.clamp(0, 1)
        self.log("Training PSNR", psnr(result, batch["gt"]), prog_bar=True)
        self.log("Training SSIM", ssim(result, batch["gt"]))  # noqa

        return loss_mse + loss_ssim

    def test_step(self, batch, batch_idx):
        # Run Deep Color Transfer
        result = self(batch).clamp(0, 1)

        # Calculate and log metrics
        self.log("Test PSNR", psnr(result, batch["gt"]), prog_bar=True)
        self.log("Test SSIM", ssim(result, batch["gt"]))  # noqa

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.estimated_stepping_batches,
            eta_min=1e-6)

        lr_scheduler_dict = {"scheduler": scheduler, "interval": "step"}

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}
