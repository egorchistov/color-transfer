# Copyright (c) 2020 LongguangWang. No License Specified
# https://github.com/The-Learning-And-Vision-Atelier-LAVA/PAM/blob/master/PASMnet/loss.py

import torch
import torch.nn.functional as F

from pasmnet.utils import warp


def masked_l1_loss(x, y, mask):
    return torch.sum(torch.abs(x - y) * mask) / torch.sum(mask)


def loss_pam_photometric(img_left, img_right, att, valid_mask):
    att_right2left, att_left2right = att
    valid_mask_left, valid_mask_right = valid_mask

    return (
        masked_l1_loss(img_left, warp(img_right, att_right2left), valid_mask_left) +
        masked_l1_loss(img_right, warp(img_left, att_left2right), valid_mask_right)
    )


def loss_pam_cycle(att_cycle, valid_mask):
    att_left2right2left, att_right2left2right = att_cycle
    valid_mask_left, valid_mask_right = valid_mask

    b, c, h, w = valid_mask[0].shape
    I = torch.eye(w, w).repeat(b, h, 1, 1).to(att_left2right2left.device)

    return (
        masked_l1_loss(att_left2right2left, I, valid_mask_left.permute(0, 2, 3, 1)) +
        masked_l1_loss(att_right2left2right, I, valid_mask_right.permute(0, 2, 3, 1))
    )


def loss_pam_smoothness(att):
    # B, H, W, W
    att_right2left, att_left2right = att

    return (
        F.l1_loss(att_right2left[:, :-1, :, :], att_right2left[:, 1:, :, :]) +
        F.l1_loss(att_right2left[:, :, :-1, :-1], att_right2left[:, :, 1:, 1:]) +
        F.l1_loss(att_left2right[:, :-1, :, :], att_left2right[:, 1:, :, :]) +
        F.l1_loss(att_left2right[:, :, :-1, :-1], att_left2right[:, :, 1:, 1:])
    )
