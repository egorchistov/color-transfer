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
            nn.BatchNorm2d(out_channels) if bn else nn.Identity(),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=not bn),
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
    def __init__(self):
        super().__init__()

        body = [nn.Conv2d(3, 64, kernel_size=3, padding=1)]
        for i in range(18):
            body.append(
                BasicBlock(64, 64, bn=False, weighted_shortcut=False)
            )
        self.body = nn.Sequential(*body)

    def forward(self, x):
        return self.body(x)


class MultiScaleFeatureExtration(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            BasicBlock(3, 16),
            BasicBlock(16, 32, stride=2),
            BasicBlock(32, 64, stride=2),
            BasicBlock(64, 96, stride=2)
        )

        self.decoder = nn.Sequential(
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
    def __init__(self, channels, weighted_shortcut=True):
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
    def __init__(self):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(64 + 64 + 1, 64, kernel_size=1),
            BasicBlock(64, 64, bn=False, weighted_shortcut=False),
            BasicBlock(64, 64, bn=False, weighted_shortcut=False),
            BasicBlock(64, 64, bn=False, weighted_shortcut=False),
            BasicBlock(64, 64, bn=False, weighted_shortcut=False),
            BasicBlock(64, 64, bn=False, weighted_shortcut=False),
            BasicBlock(64, 64, bn=False, weighted_shortcut=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )

    def forward(self, fea_left, fea_right, valid_mask):
        features = torch.cat([fea_left, fea_right, valid_mask[0]], dim=1)

        return self.body(features)


class MultiScaleTransfer(nn.Module):
    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(
            BasicBlock(2 * 64 + 1, 2 * 64 + 1, bn=False),
            BasicBlock(2 * 32 + 1, 2 * 32 + 1, bn=False),
            BasicBlock(2 * 16 + 1, 2 * 16 + 1, bn=False)
        )

        self.upsample = nn.Sequential(
            Upsample(2 * 96 + 1, 2 * 64 + 1, bn=False),
            Upsample(2 * 64 + 1, 2 * 32 + 1, bn=False),
            Upsample(2 * 32 + 1, 2 * 16 + 1, bn=False)
        )

        self.bias = nn.Conv2d(2 * 16 + 1, 3, kernel_size=1, padding=0, bias=True)

    def forward(self, fea_left, fea_right, valid_masks):
        features = [
            torch.cat([left, right, valid_mask], dim=1) for left, right, valid_mask in zip(fea_left, fea_right, valid_masks)
        ]

        x = features[3]
        x = self.decoder[0](self.upsample[0](x) + features[2])
        x = self.decoder[1](self.upsample[1](x) + features[1])
        x = self.decoder[2](self.upsample[2](x) + features[0])

        return self.bias(x)
