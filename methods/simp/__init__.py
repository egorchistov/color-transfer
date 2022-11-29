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

from methods.simp.losses import loss_pam_smoothness, loss_pam_photometric, loss_pam_cycle
from methods.simp.modules import FeatureExtration, CascadedPAM, output, Transfer


class SIMP(pl.LightningModule):
    def __init__(self, learning_rate=1e-4):
        super().__init__()
        self.learning_rate = learning_rate

        ###############################################################
        ## scale     #  1  #  1/2  #  1/4  #  1/8  #  1/16  #  1/32  ##
        ## channels  #  16 #  32   #  64   #  96   #  128   #  160   ##
        ###############################################################

        self.extraction = FeatureExtration()
        self.cas_pam = CascadedPAM()
        self.transfer = Transfer()

    def forward(self, left, right):
        b, _, h, w = left.shape

        fea_left = self.extraction(left)
        fea_right = self.extraction(right)

        costs = self.cas_pam(fea_left[-3:], fea_right[-3:])

        att_s1, att_cycle_s1, valid_mask_s1 = output(costs[0])
        att_s2, att_cycle_s2, valid_mask_s2 = output(costs[1])
        att_s3, att_cycle_s3, valid_mask_s3 = output(costs[2])

        # PAM_stage at 1/2 and 1 scales consumes too much memory
        att_s4 = [
            F.interpolate(att_s3[0].unsqueeze(1), scale_factor=2, mode="trilinear", align_corners=False).squeeze(1),
            F.interpolate(att_s3[1].unsqueeze(1), scale_factor=2, mode="trilinear", align_corners=False).squeeze(1)
        ]
        att_s5 = [
            F.interpolate(att_s4[0].unsqueeze(1), scale_factor=2, mode="trilinear", align_corners=False).squeeze(1),
            F.interpolate(att_s4[1].unsqueeze(1), scale_factor=2, mode="trilinear", align_corners=False).squeeze(1)
        ]

        # Maybe, I should add PAM_stage at 1/32 scale too
        att_s0 = [
            F.interpolate(att_s1[0].unsqueeze(1), scale_factor=0.5, mode="trilinear", align_corners=False).squeeze(1),
            F.interpolate(att_s1[1].unsqueeze(1), scale_factor=0.5, mode="trilinear", align_corners=False).squeeze(1)
        ]

        fea_warped_right = [
            torch.matmul(att[0], image.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) for image, att in zip(
                fea_right[:-3],
                [att_s5, att_s4, att_s3, att_s2, att_s1, att_s0]
            )
        ]

        corrected_left = self.transfer(fea_left[:-3], fea_warped_right)

        warped_right = torch.matmul(att_s5[0], right.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return corrected_left, (
            [att_s1, att_s2, att_s3],
            [att_cycle_s1, att_cycle_s2, att_cycle_s3],
            [valid_mask_s1, valid_mask_s2, valid_mask_s3],
            warped_right
        )

    def training_step(self, batch, batch_idx):
        left, left_gt, right = batch

        corrected_left, (att, att_cycle, valid_mask, _) = self(left, right)

        loss_PAM_P = loss_pam_photometric(left, right, att, valid_mask)
        loss_PAM_C = loss_pam_cycle(att_cycle, valid_mask)
        loss_PAM_S = loss_pam_smoothness(att)

        loss_color_correction = F.smooth_l1_loss(corrected_left, left_gt)
        loss = loss_color_correction + 0.001 * (loss_PAM_P + loss_PAM_S + loss_PAM_C)

        self.log("Photometric Loss", 0.001 * loss_PAM_P)
        self.log("Smoothness Loss", 0.001 * loss_PAM_S)
        self.log("Cycle Loss", 0.001 * loss_PAM_C)
        self.log("Color Correction Loss", loss_color_correction)

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        return [optimizer]
