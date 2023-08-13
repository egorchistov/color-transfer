from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class MultiScaleFeatureExtration(nn.Module):
    def __init__(self, layers: tuple[int, ...], channels: tuple[int, ...]):
        super().__init__()

        self.encoder = nn.Sequential(
            BasicBlock(3, channels[0]),
            BasicBlock(channels[0], channels[1], stride=2)
        )

        for scale in (2, 3, 4, 5):
            block = nn.Sequential(BasicBlock(channels[scale - 1], channels[scale], stride=2))

            for layer in range(1, layers[scale - 2]):
                block.append(BasicBlock(channels[scale], channels[scale]))

            self.encoder.append(block)

    def forward(self, x):
        features = []

        for block in self.encoder:
            x = block(x)
            features.append(x)

        return features


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


class PAM(nn.Module):
    def __init__(self, layers: int, channels: int):
        super().__init__()

        self.blocks = nn.Sequential()
        for _ in range(layers):
            self.blocks.append(PAB(channels))

    def forward(self, fea_left, fea_right, cost):
        for block in self.blocks:
            fea_left, fea_right, cost = block(fea_left, fea_right, cost)

        return fea_left, fea_right, cost


def upscale_att(atts, scale_factor=2):
    """Upscale twice matching attention maps using trilinear interpolation

    Parameters
    ----------
    atts : pair of two (B, H, W, W) tensors
        Matching attention maps: att_right2left, att_left2right

    scale_factor: float, default=2

    Returns
    -------
    x2atts : pair of two (B, Hx2, Wx2, Wx2) tensors
        Matching attention maps: att_right2left, att_left2right

    """
    x2atts = []

    for item in atts:
        item = item.unsqueeze(dim=1)  # (B, H, W, W) -> (B, 1, H, W, W)
        item = F.interpolate(item, scale_factor=scale_factor, mode="trilinear", align_corners=False)
        item = item.squeeze(dim=1)
        x2atts.append(item)

    return x2atts


class CasPAM(nn.Module):
    def __init__(self, layers: tuple[int, ...], channels: tuple[int, ...]):
        super().__init__()

        self.stages = nn.Sequential()
        for stage in range(len(layers)):
            self.stages.append(PAM(layers[-1 - stage], channels[-1 - stage]))

        self.bottlenecks = nn.Sequential()
        for stage in range(len(layers) - 1):
            self.bottlenecks.append(nn.Sequential(
                nn.Conv2d(channels[-1 - stage] + channels[-2 - stage],
                          channels[-2 - stage],
                          kernel_size=1,
                          padding=0,
                          bias=True),
                nn.LeakyReLU(inplace=True)))

    def forward(self, fea_lefts, fea_rights):
        """Apply parallax attention stages from lesser scale to greater scale

        Parameters
        ----------
        fea_lefts : feature list of - for example - scales 1/16, 1/8, 1/4
        fea_rights : feature list of - for example - scales 1/16, 1/8, 1/4

        Returns
        -------
        costs : cost list of - for example - scales 1/16, 1/8, 1/4
        """

        fea_lefts = fea_lefts[::-1]
        fea_rights = fea_rights[::-1]

        costs = []

        fea_left, fea_right, cost = self.stages[0](fea_lefts[0], fea_rights[0], cost=(0, 0))
        costs.append(cost)

        for scale in range(1, len(self.stages)):
            fea_left = F.interpolate(fea_left, scale_factor=2, mode="bilinear", align_corners=False)
            fea_right = F.interpolate(fea_right, scale_factor=2, mode="bilinear", align_corners=False)
            fea_left = self.bottlenecks[scale - 1](torch.cat([fea_left, fea_lefts[scale]], dim=1))
            fea_right = self.bottlenecks[scale - 1](torch.cat([fea_right, fea_rights[scale]], dim=1))

            cost_up = upscale_att(cost)
            fea_left, fea_right, cost = self.stages[scale](fea_left, fea_right, cost_up)
            costs.append(cost)

        atts = []
        valid_masks = []

        for cost in costs[::-1]:
            att, _, valid_mask = output(cost)
            atts.append(att)
            valid_masks.append(valid_mask)

        return atts, valid_masks


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

    # masked (lower triangular) softmax(dim=-1)
    cost_right2left = torch.tril(cost_right2left)
    cost_right2left = torch.exp(cost_right2left - cost_right2left.max(dim=-1, keepdim=True)[0])
    cost_right2left = torch.tril(cost_right2left)
    att_right2left = cost_right2left / (cost_right2left.sum(dim=-1, keepdim=True) + 1e-8)

    # masked (upper triangular) softmax(dim=-1)
    cost_left2right = torch.triu(cost_left2right)
    cost_left2right = torch.exp(cost_left2right - cost_left2right.max(dim=-1, keepdim=True)[0])
    cost_left2right = torch.triu(cost_left2right)
    att_left2right = cost_left2right / (cost_left2right.sum(dim=-1, keepdim=True) + 1e-8)

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


class Upsample(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bn=True):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=not bn),
            nn.BatchNorm2d(out_channels) if bn else nn.Identity(),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.upsample(x)


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


class MultiScaleTransfer(nn.Module):
    def __init__(self, encoder_channels, n_blocks, use_batchnorm):
        super().__init__()

        self.n_blocks = n_blocks

        self.decoder = nn.Sequential()

        for i in range(n_blocks):
            self.decoder.append(
                BasicBlock(
                    encoder_channels[-2 - i],
                    encoder_channels[-2 - i],
                    bn=use_batchnorm
                )
            )

        self.upsample = nn.Sequential()

        for i in range(n_blocks):
            self.upsample.append(
                Upsample(
                    encoder_channels[-1 - i],
                    encoder_channels[-2 - i],
                    bn=use_batchnorm
                )
            )

    def forward(self, *features):
        x = features[-1]

        for i in range(self.n_blocks):
            x = self.upsample[i](x)
            x = x + features[-2 - i]
            x = self.decoder[i](x)

        return x
