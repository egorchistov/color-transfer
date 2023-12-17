"""Deep Color Mismatch Correction in Stereoscopic 3D Images

Croci et al. employed a convolutional neural network for color-mismatch
correction. First, the network extracts features from the input stereopair.
It then feeds the extracted features into the parallax-attention mechanism,
which performs stereo matching. Matched features pass through six residual
blocks to yield the corrected stereoscopic view.

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


from __future__ import annotations


import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from piq import psnr, ssim, fsim
from kornia.losses import ssim_loss


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, bn=True, weighted_shortcut=True):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=not bn),
            nn.BatchNorm2d(out_channels) if bn else nn.Identity(),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=not bn),
            nn.BatchNorm2d(out_channels) if bn else nn.Identity()
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=not bn),
            nn.BatchNorm2d(out_channels) if bn else nn.Identity()
        ) if weighted_shortcut or in_channels != out_channels else nn.Identity()

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.shortcut(x) + self.body(x))


class FeatureExtration(nn.Module):
    def __init__(self, layers: int, channels: int):
        super().__init__()

        self.body = nn.Sequential(nn.Conv2d(3, channels, kernel_size=3, padding=1))
        for _ in range(layers):
            self.body.append(BasicBlock(channels, channels, bn=False, weighted_shortcut=False))

    def forward(self, x):
        return self.body(x)


class PAB(nn.Module):
    def __init__(self, channels: int, weighted_shortcut=True):
        super().__init__()

        self.head = BasicBlock(channels, channels, bn=False, weighted_shortcut=weighted_shortcut)
        self.query = nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=True)
        self.key = nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=True)

    def forward(self, x_left, x_right, cost):
        """Apply parallax attention to input features and update cost volume

        Parameters
        ----------
        x_left : tensor of shape (B, C, H, W)
            Features from the left image
        x_right : tensor of shape (B, C, H, W)
            Features from the right image
        cost : a pair of two (B, H, W, W) tensors
            Matching costs: cost_right2left, cost_left2righ

        Returns
        -------
        x_left : tensor of shape (B, C, H, W)
            Updated features from the left image
        x_left : tensor of shape (B, C, H, W)
            Updated features from the left image
        cost : a pair of two (B, H, W, W) tensors
            Updated matching costs: cost_right2left, cost_left2righ
        """

        b, c, h, w = x_left.shape
        fea_left = self.head(x_left)
        fea_right = self.head(x_right)

        # cost_right2left
        Q = self.query(fea_left).permute(0, 2, 3, 1)   # B * H * W * C
        K = self.key(fea_right).permute(0, 2, 1, 3)    # B * H * C * W
        cost_right2left = torch.matmul(Q, K) / c       # scale the matching cost
        cost_right2left = cost_right2left + cost[0]

        # cost_left2right
        Q = self.query(fea_right).permute(0, 2, 3, 1)  # B * H * W * C
        K = self.key(fea_left).permute(0, 2, 1, 3)     # B * H * C * W
        cost_left2right = torch.matmul(Q, K) / c       # scale the matching cost
        cost_left2right = cost_left2right + cost[1]

        return x_left + fea_left, \
            x_right + fea_right, \
            (cost_right2left, cost_left2right)


def output(costs):
    """Apply masked softmax to cost volumes and return matching attention
    maps and valid masks

    Parameters
    ----------
    costs : pair of two (B, H, W, W) tensors
        Matching costs: cost_right2left, cost_left2righ

    Returns
    -------
    atts : pair of two (B, H, W, W) tensors
        Matching attention maps: att_right2left, att_left2right
    atts_cycle : pair of two (B, H, W, W) tensors
        Matching attention cycle maps: att_left2right2left, att_right2left2right
    valid_masks : pair of two (B, 1, H, W) tensors
        Matching valid masks: valid_mask_left, valid_mask_right
    """

    cost_right2left, cost_left2right = costs

    att_right2left = F.softmax(cost_right2left, dim=-1)
    att_left2right = F.softmax(cost_left2right, dim=-1)

    # valid mask (left image)
    valid_mask_left = torch.sum(att_left2right.detach(), dim=-2) > 0.1
    valid_mask_left = valid_mask_left.unsqueeze(dim=1)

    # valid mask (right image)
    valid_mask_right = torch.sum(att_right2left.detach(), dim=-2) > 0.1
    valid_mask_right = valid_mask_right.unsqueeze(dim=1)

    # cycle-attention maps
    att_left2right2left = torch.matmul(att_right2left, att_left2right)
    att_right2left2right = torch.matmul(att_left2right, att_right2left)

    return (att_right2left, att_left2right), \
        (att_left2right2left, att_right2left2right), \
        (valid_mask_left, valid_mask_right)


def warp(image, att):
    """Warp image using matching attention map

    Parameters
    ----------
    image : (B, C, H, W) tensor
        Image to warp
    att : (B, H, W, W) tensor
        Matching attention map

    Returns
    -------
    image : (B, C, H, W) tensor
        Warped image
    """
    image = image.permute(0, 2, 3, 1)
    image = torch.matmul(att, image)  # (B, H, W, W) x (B, H, W, C) -> (B, H, W, C)
    image = image.permute(0, 3, 1, 2)

    return image


class Transfer(nn.Module):
    def __init__(self, layers: int, channels: int):
        super().__init__()

        self.body = nn.Sequential(nn.Conv2d(2 * channels + 1, channels, kernel_size=1))

        for _ in range(layers):
            self.body.append(BasicBlock(channels, channels, bn=False, weighted_shortcut=False))

        self.body.extend([
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1),
            nn.Conv2d(channels // 2, 3, kernel_size=3, padding=1)
        ])

    def forward(self, fea_left, fea_right, valid_mask):
        features = torch.cat([fea_left, fea_right, valid_mask[0]], dim=1)

        return self.body(features)


def loss_pam_photometric(img_left, img_right, att, valid_mask):
    scale = img_left.shape[2] // valid_mask[0].shape[2]

    att_right2left, att_left2right = att
    valid_mask_left, valid_mask_right = valid_mask

    img_left_scale = F.interpolate(img_left, scale_factor=1 / scale, mode="bilinear", align_corners=False)
    img_right_scale = F.interpolate(img_right, scale_factor=1 / scale, mode="bilinear", align_corners=False)

    img_right_warp = torch.matmul(att_right2left, img_right_scale.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    img_left_warp = torch.matmul(att_left2right, img_left_scale.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    loss = F.l1_loss(img_left_scale * valid_mask_left, img_right_warp * valid_mask_left) + \
        F.l1_loss(img_right_scale * valid_mask_right, img_left_warp * valid_mask_right)

    return loss


def loss_pam_cycle(att_cycle, valid_mask):
    b, c, h, w = valid_mask[0].shape
    I = torch.eye(w, w).repeat(b, h, 1, 1).to(att_cycle[0].device)

    att_left2right2left, att_right2left2right = att_cycle
    valid_mask_left, valid_mask_right = valid_mask

    loss = F.l1_loss(att_left2right2left * valid_mask_left.permute(0, 2, 3, 1),
                     I * valid_mask_left.permute(0, 2, 3, 1)) + \
        F.l1_loss(att_right2left2right * valid_mask_right.permute(0, 2, 3, 1),
                  I * valid_mask_right.permute(0, 2, 3, 1))

    return loss


def loss_pam_smoothness(att):
    att_right2left, att_left2right = att

    loss = F.l1_loss(att_right2left[:, :-1, :, :], att_right2left[:, 1:, :, :]) + \
        F.l1_loss(att_left2right[:, :-1, :, :], att_left2right[:, 1:, :, :]) + \
        F.l1_loss(att_right2left[:, :, :-1, :-1], att_right2left[:, :, 1:, 1:]) + \
        F.l1_loss(att_left2right[:, :, :-1, :-1], att_left2right[:, :, 1:, 1:])

    return loss


class DCMCS3DI(pl.LightningModule):
    def __init__(self,
                 extraction_layers=18,
                 transfer_layers=6,
                 channels=64,
                 num_logged_images=3):
        super().__init__()

        self.num_logged_images = num_logged_images

        self.extraction = FeatureExtration(layers=extraction_layers, channels=channels)
        self.pam = PAB(channels=channels, weighted_shortcut=False)
        self.value = torch.nn.Conv2d(channels, channels, kernel_size=1)
        self.transfer = Transfer(layers=transfer_layers, channels=channels)

    def forward(self, left, right):
        fea_left = self.extraction(left)
        fea_right = self.extraction(right)

        b, _, h, w = fea_left.shape

        _, _, cost = self.pam(fea_left, fea_right, cost=(0, 0))

        att, att_cycle, valid_mask = output(cost)

        fea_warped_right = warp(fea_right, att[0])

        corrected_left = self.transfer(fea_left, fea_warped_right, valid_mask)

        warped_right = warp(right, att[0])

        return corrected_left, (
            att,
            att_cycle,
            valid_mask,
            warped_right
        )

    def training_step(self, batch, batch_idx):
        left, left_gt, right = batch

        corrected_left, (att, att_cycle, valid_mask, _) = self(left, right)

        loss_l1 = F.l1_loss(corrected_left, left_gt)
        loss_mse = F.mse_loss(corrected_left, left_gt)
        loss_ssim = ssim_loss(corrected_left, left_gt, window_size=11)

        loss_pm = 0.005 * loss_pam_photometric(left, right, att, valid_mask)
        loss_cycle = 0.005 * loss_pam_cycle(att_cycle, valid_mask)
        loss_smooth = 0.0005 * loss_pam_smoothness(att)

        loss = loss_l1 + loss_mse + loss_ssim + loss_pm + loss_cycle + loss_smooth

        self.log("L1 Loss", loss_l1)
        self.log("MSE Loss", loss_mse)
        self.log("SSIM Loss", loss_ssim)

        self.log("Photometric Loss", loss_pm)
        self.log("Cycle Loss",  loss_cycle)
        self.log("Smoothness Loss",  loss_smooth)

        self.log("Loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        left, left_gt, right = batch

        corrected_left, (_, _, _, warped_right) = self(left, right)
        corrected_left = corrected_left.clamp(0, 1)

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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        return {"optimizer": optimizer}