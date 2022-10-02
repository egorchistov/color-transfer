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
    --max_epochs=100
```

We use Tesla P100-16GB in our expreriments.

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
from torchvision.transforms.functional import normalize

from methods.simp.losses import warp_disp
from methods.simp.losses import loss_pam_smoothness, loss_pam_photometric, loss_pam_cycle, loss_disp_smoothness, \
    loss_disp_unsupervised
from methods.simp.modules import Hourglass, CascadedPAM, Output, ColorCorrection


class SIMP(pl.LightningModule):
    def __init__(self, learning_rate=1e-4):
        super().__init__()
        self.learning_rate = learning_rate

        ###############################################################
        ## scale     #  1  #  1/2  #  1/4  #  1/8  #  1/16  #  1/32  ##
        ## channels  #  16 #  32   #  64   #  96   #  128   #  160   ##
        ###############################################################

        self.pam_hourglass = Hourglass([32, 64, 96, 128, 160])
        self.color_hourglass = Hourglass([32, 64, 96, 128, 160])
        self.cas_pam = CascadedPAM([128, 96, 64])
        self.output = Output()
        self.color_correction = ColorCorrection([64, 96, 128, 160, 160, 128, 96, 64, 32, 16])

    def forward(self, left, right, max_disp=0):
        b, _, h, w = left.shape

        normalize(left, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True)
        normalize(right, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True)

        (fea_left_s1, fea_left_s2, fea_left_s3), fea_refine = self.pam_hourglass(left)
        (fea_right_s1, fea_right_s2, fea_right_s3), _ = self.pam_hourglass(right)

        cost_s1, cost_s2, cost_s3 = self.cas_pam([fea_left_s1, fea_left_s2, fea_left_s3],
                                                 [fea_right_s1, fea_right_s2, fea_right_s3])

        disp_s1, att_s1, att_cycle_s1, valid_mask_s1 = self.output(cost_s1, max_disp // 16)
        disp_s2, att_s2, att_cycle_s2, valid_mask_s2 = self.output(cost_s2, max_disp // 8)
        disp_s3, att_s3, att_cycle_s3, valid_mask_s3 = self.output(cost_s3, max_disp // 4)

        (color_fea_left_s1, color_fea_left_s2, color_fea_left_s3) = self.color_hourglass(left)[0]
        (color_fea_right_s1, color_fea_right_s2, color_fea_right_s3) = self.color_hourglass(right)[0]

        warped_color_fea_right_s1 = warp_disp(color_fea_right_s1, -disp_s1)  # scale: 1/16
        warped_color_fea_right_s2 = warp_disp(color_fea_right_s2, -disp_s2)  # scale: 1/8
        warped_color_fea_right_s3 = warp_disp(color_fea_right_s3, -disp_s3)  # scale: 1/4

        corrected_left = self.color_correction(
            (color_fea_left_s1, color_fea_left_s2, color_fea_left_s3),
            (warped_color_fea_right_s1, warped_color_fea_right_s2, warped_color_fea_right_s3),
            (valid_mask_s1, valid_mask_s2, valid_mask_s3),
            left)

        normalize(corrected_left, mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255], std=[1/0.229, 1/0.224, 1/0.255], inplace=True)

        return corrected_left, (
            4 * F.interpolate(disp_s3, scale_factor=4, mode="nearest"),
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
        loss = loss_color_correction + loss_P + 0.1 * loss_S + loss_PAM_P + loss_PAM_S + loss_PAM_C

        self.log("Photometric Loss", loss_PAM_P + loss_P)
        self.log("Smoothness Loss", 0.1 * loss_S + loss_PAM_S)
        self.log("Cycle Loss", loss_PAM_C)
        self.log("Color Correction Loss", loss_color_correction)
        self.log("Loss", loss)

        if batch_idx == 0 and isinstance(self.logger, WandbLogger):
            self.logger.log_image(
                key="Train",
                images=[left, warp_disp(right, -disp), corrected_left.clamp(0, 1), left_gt, right, disp, F.interpolate(valid_mask[-1][0], scale_factor=4, mode="nearest")],
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
                images=[left, warp_disp(right, -disp), corrected_left.clamp(0, 1), left_gt, right, disp, F.interpolate(valid_mask[-1][0], scale_factor=4, mode="nearest")],
                caption=["Left Distorted", "Warped Right", "Left Corrected", "Left", "Right", "Disparity",
                         "Valid Mask"])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        return {"optimizer": optimizer}
