import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import morphology


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, bn=True):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=not bn),
            nn.BatchNorm2d(out_channels) if bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=not bn),
            nn.BatchNorm2d(out_channels) if bn else nn.Identity()
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=not bn),
            nn.BatchNorm2d(out_channels) if bn else nn.Identity()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.shortcut(x) + self.body(x))


class FeatureExtration(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            BasicBlock(3, 16),
            BasicBlock(16, 32, stride=2),
            BasicBlock(32, 64, stride=2),
            BasicBlock(64, 96, stride=2),
            BasicBlock(96, 128, stride=2),
            BasicBlock(128, 160, stride=2)
        )

        self.decoder = nn.Sequential(
            nn.Sequential(
                Upsample(160, 128),
                BasicBlock(128, 128),
            ),
            nn.Sequential(
                Upsample(128, 96),
                BasicBlock(96, 96),
            ),
            nn.Sequential(
                Upsample(96, 64),
                BasicBlock(64, 64),
            )
        )

    def forward(self, x):
        features = []

        for layer in self.encoder:
            x = layer(x)
            features.append(x)

        for layer in self.decoder:
            x = layer(x)
            features.append(x)

        return features


class PAB(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

        self.query = nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=True)
        self.key = nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=True)

    def forward(self, x_left, x_right, cost):
        """
        :param x_left:      features from the left image  (B * C * H * W)
        :param x_right:     features from the right image (B * C * H * W)
        :param cost:        input matching cost           (B * H * W * W)
        """

        b, c, h, w = x_left.shape
        fea_left = self.head(x_left)
        fea_right = self.head(x_right)

        # C_right2left
        Q = self.query(fea_left).permute(0, 2, 3, 1).contiguous()                     # B * H * W * C
        K = self.key(fea_right).permute(0, 2, 1, 3) .contiguous()                     # B * H * C * W
        cost_right2left = torch.matmul(Q, K) / c                                      # scale the matching cost
        cost_right2left = cost_right2left + cost[0]

        # C_left2right
        Q = self.query(fea_right).permute(0, 2, 3, 1).contiguous()                    # B * H * W * C
        K = self.key(fea_left).permute(0, 2, 1, 3).contiguous()                       # B * H * C * W
        cost_left2right = torch.matmul(Q, K) / c                                      # scale the matching cost
        cost_left2right = cost_left2right + cost[1]

        return x_left + fea_left, \
            x_right + fea_right, \
            (cost_right2left, cost_left2right)


class StagePAM(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.blocks = nn.Sequential(
            PAB(channels),
            PAB(channels),
            PAB(channels),
            PAB(channels)
        )

    def forward(self, fea_left, fea_right, cost):
        for block in self.blocks:
            fea_left, fea_right, cost = block(fea_left, fea_right, cost)

        return fea_left, fea_right, cost


class CascadedPAM(nn.Module):
    def __init__(self):
        super().__init__()

        self.stages = nn.Sequential(
            StagePAM(128),
            StagePAM(96),
            StagePAM(64)
        )

        # bottleneck in stage 2
        self.b2 = nn.Sequential(
            nn.Conv2d(128 + 96, 96, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )

        # bottleneck in stage 3
        self.b3 = nn.Sequential(
            nn.Conv2d(96 + 64, 64, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, fea_left, fea_right):
        """
        :param fea_left:    feature list [fea_left_s1, fea_left_s2, fea_left_s3]
        :param fea_right:   feature list [fea_right_s1, fea_right_s2, fea_right_s3]
        """
        fea_left_s1, fea_left_s2, fea_left_s3 = fea_left
        fea_right_s1, fea_right_s2, fea_right_s3 = fea_right

        b, _, h_s1, w_s1 = fea_left_s1.shape
        b, _, h_s2, w_s2 = fea_left_s2.shape

        # stage 1: 1/16
        cost_s0 = [
            torch.zeros(b, h_s1, w_s1, w_s1).to(fea_right_s1.device),
            torch.zeros(b, h_s1, w_s1, w_s1).to(fea_right_s1.device)
        ]

        fea_left, fea_right, cost_s1 = self.stages[0](fea_left_s1, fea_right_s1, cost_s0)

        # stage 2: 1/8
        fea_left = F.interpolate(fea_left, scale_factor=2, mode="bilinear", align_corners=False)
        fea_right = F.interpolate(fea_right, scale_factor=2, mode="bilinear", align_corners=False)
        fea_left = self.b2(torch.cat([fea_left, fea_left_s2], dim=1))
        fea_right = self.b2(torch.cat([fea_right, fea_right_s2], dim=1))

        cost_s1_up = [
            F.interpolate(cost_s1[0].unsqueeze(1), scale_factor=2, mode="trilinear", align_corners=False).squeeze(1),
            F.interpolate(cost_s1[1].unsqueeze(1), scale_factor=2, mode="trilinear", align_corners=False).squeeze(1)
        ]

        fea_left, fea_right, cost_s2 = self.stages[1](fea_left, fea_right, cost_s1_up)

        # stage 3: 1/4
        fea_left = F.interpolate(fea_left, scale_factor=2, mode="bilinear", align_corners=False)
        fea_right = F.interpolate(fea_right, scale_factor=2, mode="bilinear", align_corners=False)
        fea_left = self.b3(torch.cat([fea_left, fea_left_s3], dim=1))
        fea_right = self.b3(torch.cat([fea_right, fea_right_s3], dim=1))

        cost_s2_up = [
            F.interpolate(cost_s2[0].unsqueeze(1), scale_factor=2, mode="trilinear", align_corners=False).squeeze(1),
            F.interpolate(cost_s2[1].unsqueeze(1), scale_factor=2, mode="trilinear", align_corners=False).squeeze(1)
        ]

        fea_left, fea_right, cost_s3 = self.stages[2](fea_left, fea_right, cost_s2_up)

        return [cost_s1, cost_s2, cost_s3]


class Output(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, cost):
        cost_right2left, cost_left2right = cost
        b, h, w, _ = cost_right2left.shape

        # M_right2left
        # exclude negative disparities
        cost_right2left = torch.tril(cost_right2left)
        cost_right2left = torch.exp(cost_right2left - cost_right2left.max(dim=-1, keepdim=True)[0])
        cost_right2left = torch.tril(cost_right2left)

        att_right2left = cost_right2left / (cost_right2left.sum(dim=-1, keepdim=True) + 1e-8)

        # M_left2right
        # exclude negative disparities
        cost_left2right = torch.triu(cost_left2right)
        cost_left2right = torch.exp(cost_left2right - cost_left2right.max(dim=-1, keepdim=True)[0])
        cost_left2right = torch.triu(cost_left2right)

        att_left2right = cost_left2right / (cost_left2right.sum(dim=-1, keepdim=True) + 1e-8)

        # valid mask (left image)
        valid_mask_left = torch.sum(att_left2right.detach(), dim=-2) > 0.1
        valid_mask_left = valid_mask_left.view(b, 1, h, w)

        # valid mask (right image)
        valid_mask_right = torch.sum(att_right2left.detach(), dim=-2) > 0.1
        valid_mask_right = valid_mask_right.view(b, 1, h, w)

        # cycle-attention maps
        att_left2right2left = torch.matmul(att_right2left, att_left2right).view(b, h, w, w)
        att_right2left2right = torch.matmul(att_left2right, att_right2left).view(b, h, w, w)

        return (att_right2left.view(b, h, w, w), att_left2right.view(b, h, w, w)), \
            (att_left2right2left, att_right2left2right), \
            (valid_mask_left, valid_mask_right)


class Upsample(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bn=True):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=not bn),
            nn.BatchNorm2d(out_channels) if bn else nn.Identity(),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.upsample(x)


class Transfer(nn.Module):
    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(
            BasicBlock(2 * 128, 2 * 128, bn=False),
            BasicBlock(2 * 96, 2 * 96, bn=False),
            BasicBlock(2 * 64, 2 * 64, bn=False),
            BasicBlock(2 * 32, 2 * 32, bn=False),
            BasicBlock(2 * 16, 2 * 16, bn=False)
        )

        self.upsample = nn.Sequential(
            Upsample(2 * 160, 2 * 128, bn=False),
            Upsample(2 * 128, 2 * 96, bn=False),
            Upsample(2 * 96, 2 * 64, bn=False),
            Upsample(2 * 64, 2 * 32, bn=False),
            Upsample(2 * 32, 2 * 16, bn=False)
        )

        self.bias = nn.Conv2d(2 * 16, 3, kernel_size=1, padding=0, bias=True)

    def forward(self, fea_left, fea_right):
        features = [
            torch.cat([left, right], dim=1) for left, right in zip(fea_left, fea_right)
        ]

        x = features[5]
        x = self.decoder[0](self.upsample[0](x) + features[4])
        x = self.decoder[1](self.upsample[1](x) + features[3])
        x = self.decoder[2](self.upsample[2](x) + features[2])
        x = self.decoder[3](self.upsample[3](x) + features[1])
        x = self.decoder[4](self.upsample[4](x) + features[0])

        return self.bias(x)
