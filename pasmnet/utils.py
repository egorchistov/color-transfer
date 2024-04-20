# Copyright (c) 2020 LongguangWang. No License Specified
# https://github.com/The-Learning-And-Vision-Atelier-LAVA/PAM/blob/master/PASMnet/models/modules.py

import torch
import torch.nn.functional as F


def output(costs):
    """Apply masked softmax to cost volumes and return matching attention
    maps and valid masks

    Parameters
    ----------
    costs : a pair of two (B, H, W, W) tensors
        Matching costs: cost_right2left, cost_left2righ

    Returns
    -------
    atts : a pair of two (B, H, W, W) tensors
        Matching attention maps: att_right2left, att_left2right
    atts_cycle : a pair of two (B, H, W, W) tensors
        Matching attention cycle maps: att_left2right2left, att_right2left2right
    valid_masks : a pair of two (B, 1, H, W) tensors
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


def regress_disp(att, valid_mask):
    """Regress disparity from matching attention map

    Parameters
    ----------
    att : (B, H, W, W) tensor
        Matching attention map
    valid_mask : (B, 1, H, W) tensor
        Matching valid mask

    Returns
    -------
    disp : (B, 1, H, W) tensor
        Disparity regressed from matching attention map
    """

    b, h, w, _ = att.shape
    index = torch.arange(w, device=att.device).reshape(1, 1, 1, w).float()
    disp_ini = index - torch.sum(att * index, dim=-1).reshape(b, 1, h, w)

    # partial conv
    filter1 = torch.zeros(1, 3, device=att.device)
    filter1[0, 0] = 1
    filter1[0, 1] = 1
    filter1 = filter1.reshape(1, 1, 1, 3)

    filter2 = torch.zeros(1, 3, device=att.device)
    filter2[0, 1] = 1
    filter2[0, 2] = 1
    filter2 = filter2.reshape(1, 1, 1, 3)

    valid_mask_0 = valid_mask
    disp = disp_ini * valid_mask_0

    valid_mask_num = 1
    while valid_mask_num > 0:
        valid_mask_1 = F.conv2d(valid_mask_0, filter1, padding=[0, 1])
        disp = disp * valid_mask_0 + \
            F.conv2d(disp, filter1, padding=[0, 1]) / (valid_mask_1 + 1e-4) * ((valid_mask_1 > 0).float() - valid_mask_0)
        valid_mask_num = (valid_mask_1 > 0).float().sum() - valid_mask_0.sum()
        valid_mask_0 = (valid_mask_1 > 0).float()

    valid_mask_num = 1
    while valid_mask_num > 0:
        valid_mask_1 = F.conv2d(valid_mask_0, filter2, padding=[0, 1])
        disp = disp * valid_mask_0 + \
            F.conv2d(disp, filter2, padding=[0, 1]) / (valid_mask_1 + 1e-4) * ((valid_mask_1 > 0).float() - valid_mask_0)
        valid_mask_num = (valid_mask_1 > 0).float().sum() - valid_mask_0.sum()
        valid_mask_0 = (valid_mask_1 > 0).float()

    return disp_ini * valid_mask + disp * (1 - valid_mask)


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
