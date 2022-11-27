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
from methods.simp.modules import FeatureExtration, CascadedPAM, Output, ColorCorrection


class SIMP(pl.LightningModule):
    def __init__(self, learning_rate=1e-4):
        super().__init__()
        self.learning_rate = learning_rate

        ###############################################################
        ## scale     #  1  #  1/2  #  1/4  #  1/8  #  1/16  #  1/32  ##
        ## channels  #  16 #  32   #  64   #  96   #  128   #  160   ##
        ###############################################################

        self.extraction = FeatureExtration()
        self.cas_pam = CascadedPAM([128, 96, 64])
        self.output = Output()
        self.color_correction = ColorCorrection([16, 32, 64, 96, 128, 160])

    def forward(self, left, right, max_disp=0):
        b, _, h, w = left.shape

        fea_left = self.extraction(left)
        fea_right = self.extraction(right)

        costs = self.cas_pam(fea_left[-3:], fea_right[-3:])

        disp_s1, att_s1, att_cycle_s1, valid_mask_s1 = self.output(costs[0], max_disp // 16)
        disp_s2, att_s2, att_cycle_s2, valid_mask_s2 = self.output(costs[1], max_disp // 8)
        disp_s3, att_s3, att_cycle_s3, valid_mask_s3 = self.output(costs[2], max_disp // 4)

        # PAM_stage at 1/2 and 1 scales consumes too much memory
        disp_s4 = 2 * F.interpolate(disp_s3, scale_factor=2, mode="bilinear", align_corners=False)
        disp_s5 = 2 * F.interpolate(disp_s4, scale_factor=2, mode="bilinear", align_corners=False)

        # Maybe, I should add PAM_stage at 1/32 scale too
        disp_s0 = 0.5 * F.interpolate(disp_s1, scale_factor=0.5, mode="bilinear", align_corners=False)

        fea_warped_right = [
            warp_disp(image.detach(), -disp.detach()) for image, disp in zip(
                fea_right[:-3],
                [disp_s5, disp_s4, disp_s3, disp_s2, disp_s1, disp_s0]
            )
        ]

        corrected_left = self.color_correction(fea_left[:-3], fea_warped_right)

        return corrected_left, (
            disp_s5,
            [att_s1, att_s2, att_s3],
            [att_cycle_s1, att_cycle_s2, att_cycle_s3],
            [valid_mask_s1, valid_mask_s2, valid_mask_s3]
        )

    def training_step(self, batch, batch_idx):
        left, left_gt, right = batch

        corrected_left, (disp, att, att_cycle, valid_mask) = self(left, right)

        loss_P = loss_disp_unsupervised(left, right, disp, F.interpolate(valid_mask[-1][0], scale_factor=4, mode="nearest"))
        loss_S = loss_disp_smoothness(disp, left)
        loss_PAM_P = loss_pam_photometric(left, right, att, valid_mask)
        loss_PAM_C = loss_pam_cycle(att_cycle, valid_mask)
        loss_PAM_S = loss_pam_smoothness(att)

        loss_color_correction = F.smooth_l1_loss(corrected_left, left_gt)
        loss = loss_color_correction + 0.001 * (loss_P + 0.1 * loss_S + loss_PAM_P + loss_PAM_S + loss_PAM_C)

        self.log("Photometric Loss", 0.001 * (loss_PAM_P + loss_P))
        self.log("Smoothness Loss", 0.001 * (0.1 * loss_S + loss_PAM_S))
        self.log("Cycle Loss", 0.001 * loss_PAM_C)
        self.log("Color Correction Loss", loss_color_correction)

        self.log("Loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        left, left_gt, right = batch

        corrected_left, (disp, _, _, valid_mask) = self(left, right)
        valid_mask = F.interpolate(valid_mask[-1][0], scale_factor=4, mode="nearest")

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
