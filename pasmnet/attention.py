# Copyright (c) 2020 LongguangWang. No License Specified
# https://github.com/The-Learning-And-Vision-Atelier-LAVA/PAM/blob/master/PASMnet/models/modules.py

import torch

from pasmnet.backbone import ResB


class PAB(torch.nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.head = ResB(channels, channels)
        self.query = torch.nn.Conv2d(channels, channels, kernel_size=1)
        self.key = torch.nn.Conv2d(channels, channels, kernel_size=1)
        self.value = torch.nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x_left, x_right):
        """Apply parallax attention to input features and compute the cost volume

        Parameters
        ----------
        x_left : tensor of shape (B, C, H, W)
            Features from the left image
        x_right : tensor of shape (B, C, H, W)
            Features from the right image

        Returns
        -------
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

        # cost_left2right
        Q = self.query(fea_right).permute(0, 2, 3, 1)  # B * H * W * C
        K = self.key(fea_left).permute(0, 2, 1, 3)     # B * H * C * W
        cost_left2right = torch.matmul(Q, K) / c       # scale the matching cost

        return cost_right2left, cost_left2right
