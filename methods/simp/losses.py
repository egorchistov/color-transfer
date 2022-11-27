import torch
import torch.nn as nn
import torch.nn.functional as F


def loss_pam_photometric(img_left, img_right, att, valid_mask, mask=None):
    weight = [0.2, 0.3, 0.5]
    loss = torch.zeros(1).to(img_left.device)

    for idx_scale in range(len(att)):
        scale = img_left.size()[2] // valid_mask[idx_scale][0].size()[2]

        att_right2left = att[idx_scale][0]                          # b * h * w * w
        att_left2right = att[idx_scale][1]
        valid_mask_left = valid_mask[idx_scale][0]                  # b * 1 * h * w
        valid_mask_right = valid_mask[idx_scale][1]

        if mask is not None:
            valid_mask_left = valid_mask_left * (nn.AvgPool2d(scale)(mask[0].float()) > 0).float()
            valid_mask_right = valid_mask_right * (nn.AvgPool2d(scale)(mask[1].float()) > 0).float()

        img_left_scale  = F.interpolate(img_left,  scale_factor=1/scale, mode='bilinear')
        img_right_scale = F.interpolate(img_right, scale_factor=1/scale, mode='bilinear')

        img_right_warp = torch.matmul(att_right2left, img_right_scale.permute(0, 2, 3, 1).contiguous())
        img_right_warp = img_right_warp.permute(0, 3, 1, 2)
        img_left_warp = torch.matmul(att_left2right,  img_left_scale.permute(0, 2, 3, 1).contiguous())
        img_left_warp = img_left_warp.permute(0, 3, 1, 2)

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

        att_left2right2left = att_cycle[idx_scale][0]
        att_right2left2right = att_cycle[idx_scale][1]
        valid_mask_left = valid_mask[idx_scale][0]
        valid_mask_right = valid_mask[idx_scale][1]

        loss_scale = F.l1_loss(att_left2right2left * valid_mask_left.permute(0, 2, 3, 1), I * valid_mask_left.permute(0, 2, 3, 1)) + \
            F.l1_loss(att_right2left2right * valid_mask_right.permute(0, 2, 3, 1), I * valid_mask_right.permute(0, 2, 3, 1))

        loss = loss + weight[idx_scale] * loss_scale

    return loss


def loss_pam_smoothness(att):
    weight = [0.2, 0.3, 0.5]
    loss = torch.zeros(1).to(att[0][0].device)

    for idx_scale in range(len(att)):
        att_right2left = att[idx_scale][0]
        att_left2right = att[idx_scale][1]

        loss_scale = F.l1_loss(att_right2left[:, :-1, :, :], att_right2left[:, 1:, :, :]) + \
            F.l1_loss(att_left2right[:, :-1, :, :], att_left2right[:, 1:, :, :]) + \
            F.l1_loss(att_right2left[:, :, :-1, :-1], att_right2left[:, :, 1:, 1:]) + \
            F.l1_loss(att_left2right[:, :, :-1, :-1], att_left2right[:, :, 1:, 1:])

        loss = loss + weight[idx_scale] * loss_scale

    return loss
