import torch
import torch.nn as nn
import torch.nn.functional as F


def loss_pam_photometric(img_left, img_right, att, valid_mask):
    weight = [0.2, 0.3, 0.5]
    loss = torch.zeros(1).to(img_left.device)

    for idx_scale in range(len(att)):
        scale = img_left.shape[2] // valid_mask[idx_scale][0].shape[2]

        att_right2left, att_left2right = att[idx_scale]
        valid_mask_left, valid_mask_right = valid_mask[idx_scale]

        img_left_scale  = F.interpolate(img_left,  scale_factor=1 / scale, mode="bilinear", align_corners=False)
        img_right_scale = F.interpolate(img_right, scale_factor=1 / scale, mode="bilinear", align_corners=False)

        img_right_warp = torch.matmul(att_right2left, img_right_scale.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        img_left_warp = torch.matmul(att_left2right, img_left_scale.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        loss_scale = F.l1_loss(img_left_scale * valid_mask_left, img_right_warp * valid_mask_left) + \
            F.l1_loss(img_right_scale * valid_mask_right, img_left_warp * valid_mask_right)

        loss = loss + weight[idx_scale] * loss_scale

    return loss


def loss_pam_cycle(att_cycle, valid_mask):
    weight = [0.2, 0.3, 0.5]
    loss = torch.zeros(1).to(att_cycle[0][0].device)

    for idx_scale in range(len(att_cycle)):
        b, c, h, w = valid_mask[idx_scale][0].shape
        I = torch.eye(w, w).repeat(b, h, 1, 1).to(att_cycle[0][0].device)

        att_left2right2left, att_right2left2right = att_cycle[idx_scale]
        valid_mask_left, valid_mask_right = valid_mask[idx_scale]

        loss_scale = F.l1_loss(att_left2right2left * valid_mask_left.permute(0, 2, 3, 1), I * valid_mask_left.permute(0, 2, 3, 1)) + \
            F.l1_loss(att_right2left2right * valid_mask_right.permute(0, 2, 3, 1), I * valid_mask_right.permute(0, 2, 3, 1))

        loss = loss + weight[idx_scale] * loss_scale

    return loss


def loss_pam_smoothness(att):
    weight = [0.2, 0.3, 0.5]
    loss = torch.zeros(1).to(att[0][0].device)

    for idx_scale in range(len(att)):
        att_right2left, att_left2right = att[idx_scale]

        loss_scale = F.l1_loss(att_right2left[:, :-1, :, :], att_right2left[:, 1:, :, :]) + \
            F.l1_loss(att_left2right[:, :-1, :, :], att_left2right[:, 1:, :, :]) + \
            F.l1_loss(att_right2left[:, :, :-1, :-1], att_right2left[:, :, 1:, 1:]) + \
            F.l1_loss(att_left2right[:, :, :-1, :-1], att_left2right[:, :, 1:, 1:])

        loss = loss + weight[idx_scale] * loss_scale

    return loss
