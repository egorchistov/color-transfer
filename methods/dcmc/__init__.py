"""Deep Color Mismatch Correction in Stereoscopic 3D Images

Follow dataset preparation instructions in datasets package.
Then use this command to start training

```shell
python -m methods.dcmc.train \
    --dataset_path=datasets/dataset \
    --accelerator="gpu" \
    --img_height=96 \
    --img_width=192 \
    --batch_size=16 \
    --max_epochs=100
```

We use Tesla P100-16GB in our expreriments.
In this implementation we do not use horizontal and vertical flips as augmentation.

See https://wandb.ai/egorchistov/dcmc for training logs and artifacts.

Links
-----
https://github.com/The-Learning-And-Vision-Atelier-LAVA/PAM
"""

from pathlib import Path

import wandb
import torch
import pytorch_lightning as pl
from kornia import image_to_tensor, tensor_to_image
from torch import nn
from kornia.metrics import psnr, ssim
from kornia.losses import ssim_loss
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger

from methods.dcmc.modules import Encoder, PASM
from methods.dcmc.losses import warp_disp
from methods.dcmc.losses import loss_pam_smoothness, loss_pam_photometric, loss_pam_cycle, loss_disp_smoothness, \
    loss_disp_unsupervised


def deep_color_mismatch_correction(target, reference):
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

    run = wandb.init()
    artifact = run.use_artifact("egorchistov/dcmc/model-3e2qi9lr:v100", type="model")
    artifact_dir = artifact.download()
    model = DCMC.load_from_checkpoint(Path(artifact_dir).resolve() / "model.ckpt")

    target = image_to_tensor(target, keepdim=False).float()
    reference = image_to_tensor(reference, keepdim=False).float()

    model.eval()
    with torch.no_grad():
        corrected_left, _ = model(target, reference)

    return tensor_to_image(corrected_left)


class DCMC(pl.LightningModule):
    def __init__(self, learning_rate=1e-4):
        super().__init__()
        self.learning_rate = learning_rate

        self.feature_extraction = Encoder(n_blocks=18, channels_in=3, channels_out=64)
        self.pam = PASM(channels=64)

        # Concatenate left and right features and valid mask: 64 + 64 + 1
        self.color_correction = torch.nn.Sequential(
            Encoder(n_blocks=6, channels_in=64 + 64 + 1, channels_out=64),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 3, kernel_size=3, padding=1))

    def forward(self, left, right):
        left_features = self.feature_extraction(left)
        right_features = self.feature_extraction(right)

        disp, att, att_cycle, valid_mask = self.pam(left_features, right_features)

        warped_right_features = warp_disp(right_features, -disp)

        x = torch.cat([left_features, warped_right_features, valid_mask[0]], dim=1)
        corrected_left = self.color_correction(x)

        return corrected_left, (disp, att, att_cycle, valid_mask)

    def training_step(self, batch, batch_idx):
        left, left_gt, right = batch

        corrected_left, (disp, att, att_cycle, valid_mask) = self(left, right)

        loss_P = loss_disp_unsupervised(left, right, disp, valid_mask[0])
        loss_S = loss_disp_smoothness(disp, left)
        loss_PAM_P = loss_pam_photometric(left, right, att, valid_mask)
        loss_PAM_C = loss_pam_cycle(att_cycle, valid_mask)
        loss_PAM_S = loss_pam_smoothness(att)
        loss_color_correction = F.l1_loss(corrected_left, left_gt) + \
            F.mse_loss(corrected_left, left_gt) + \
            ssim_loss(corrected_left, left_gt, window_size=11)
        loss = loss_color_correction + 0.005 * (loss_P + 0.1 * loss_S + loss_PAM_P + loss_PAM_S + loss_PAM_C)

        self.log("Photometric Loss", 0.005 * (loss_PAM_P + loss_P))
        self.log("Smoothness Loss",  0.005 * (0.1 * loss_S + loss_PAM_S))
        self.log("Cycle Loss",  0.005 * loss_PAM_C)
        self.log("Color Correction Loss", loss_color_correction)
        self.log("Loss", loss)

        if batch_idx == 0 and isinstance(self.logger, WandbLogger):
            self.logger.log_image(
                key="Train",
                images=[left, warp_disp(right, -disp), corrected_left.clamp(0, 1), left_gt, right, disp, valid_mask[0]],
                caption=["Left Distorted", "Warped Right", "Left Corrected", "Left", "Right", "Disparity",
                         "Valid Mask"])

        return loss

    def validation_step(self, batch, batch_idx):
        left, left_gt, right = batch

        corrected_left, (disp, _, _, valid_mask) = self(left, right)

        psnr_value = psnr(corrected_left, left_gt, max_val=1)
        ssim_value = ssim(corrected_left, left_gt, window_size=11).mean()

        self.log("PSNR", psnr_value)
        self.log("SSIM", ssim_value)

        if batch_idx == 0 and isinstance(self.logger, WandbLogger):
            self.logger.log_image(
                key="Validation",
                images=[left, warp_disp(right, -disp), corrected_left.clamp(0, 1), left_gt, right, disp, valid_mask[0]],
                caption=["Left Distorted", "Warped Right", "Left Corrected", "Left", "Right", "Disparity",
                         "Valid Mask"])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        return {"optimizer": optimizer}
