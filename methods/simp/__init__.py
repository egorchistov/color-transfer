"""Stereo Investigation Model Precise

Follow dataset preparation instructions in datasets package.
Then use this command to start training

```shell
python -m methods.simp.train \
    --dataset_path=datasets/dataset \
    --accelerator="gpu" \
    --img_height=256 \
    --img_width=512  \
    --batch_size=16  \
    --max_epochs=100  \
    --num_workers=2  \
    --check_val_every_n_epoch=5
```

In our expriments we use Kaggle GPU environment:
Intel Xeon CPU (2 cores), 13 Gb RAM and Tesla P100-16GB.

See https://wandb.ai/egorchistov/simp for training logs and artifacts.

Links
-----
https://github.com/The-Learning-And-Vision-Atelier-LAVA/PAM
"""

import torch
import pytorch_lightning as pl
from kornia.metrics import psnr, ssim
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger

from methods.simp.losses import warp_disp
from methods.simp.losses import loss_pam_smoothness, loss_pam_photometric, loss_pam_cycle, loss_disp_smoothness, \
    loss_disp_unsupervised
from methods.simp.modules import Encoder, PAM, Hourglass, output


class SIMP(pl.LightningModule):
    def __init__(self, learning_rate=1e-4):
        super().__init__()
        self.learning_rate = learning_rate

        ####################################
        ## scale     #  1  #  1/2  #  1/4 ##
        ## channels  #  32 #  64   #  96  ##
        ####################################

        self.feature_extractor = Encoder([3, 32, 64, 96], bn=True)
        self.correlation = PAM(96, bn=True)
        self.color_correction = Hourglass([6, 32, 64, 96, 3], bn=False)

    def forward(self, left, right):
        b, _, h, w = left.shape

        left_features = self.feature_extractor(left)
        right_features = self.feature_extractor(right)

        cost_volume = self.correlation(left_features, right_features)
        disp, att, att_cycle, valid_mask = output(cost_volume)

        disp = 4 * F.interpolate(disp, scale_factor=4)
        warped_right = warp_disp(right, -disp)

        corrected_left = self.color_correction(torch.cat([left, warped_right], dim=1))

        return corrected_left, (disp, att, att_cycle, valid_mask)

    def training_step(self, batch, batch_idx):
        left, left_gt, right = batch

        corrected_left, (disp, att, att_cycle, valid_mask) = self(left, right)

        loss_P = loss_disp_unsupervised(left, right, disp, F.interpolate(valid_mask[0], scale_factor=4))
        loss_S = loss_disp_smoothness(disp, left)
        loss_PAM_P = loss_pam_photometric(left, right, att, valid_mask)
        loss_PAM_C = loss_pam_cycle(att_cycle, valid_mask)
        loss_PAM_S = loss_pam_smoothness(att)

        loss_color_correction = F.smooth_l1_loss(corrected_left, left_gt)
        loss = loss_color_correction + 0.005 * (loss_P + 0.1 * loss_S + loss_PAM_P + loss_PAM_S + loss_PAM_C)

        self.log("Photometric Loss", 0.005 * (loss_PAM_P + loss_P))
        self.log("Smoothness Loss", 0.005 * (0.1 * loss_S + loss_PAM_S))
        self.log("Cycle Loss", 0.005 * loss_PAM_C)
        self.log("Color Correction Loss", loss_color_correction)

        self.log("Loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        left, left_gt, right = batch

        corrected_left, (disp, _, _, valid_mask) = self(left, right)
        valid_mask = F.interpolate(valid_mask[0], scale_factor=4)

        psnr_value = psnr(corrected_left, left_gt, max_val=1)
        ssim_value = ssim(corrected_left, left_gt, window_size=11).mean()

        self.log("PSNR", psnr_value)
        self.log("SSIM", ssim_value)

        if batch_idx == 0 and isinstance(self.logger, WandbLogger):
            self.logger.log_image(
                key="Validation",
                images=[left, warp_disp(right, -disp), corrected_left.clamp(0, 1), left_gt, right, disp, valid_mask],
                caption=["Left Distorted", "Warped Right", "Left Corrected", "Left", "Right", "Disparity",
                         "Valid Mask"])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        return [optimizer]
