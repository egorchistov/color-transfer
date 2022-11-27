import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import morphology
import pytorch_lightning as pl


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, scale_factor=1):
        super().__init__()
        if scale_factor != 1:
            self.upsample = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=False)
        else:
            self.upsample = None

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)

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
            BasicBlock(160, 128, scale_factor=2),
            BasicBlock(128, 96, scale_factor=2),
            BasicBlock(96, 64, scale_factor=2)
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
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.query = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.key = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

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


class PAM_stage(nn.Module):
    def __init__(self, channels):
        super(PAM_stage, self).__init__()
        self.pab1 = PAB(channels)
        self.pab2 = PAB(channels)
        self.pab3 = PAB(channels)
        self.pab4 = PAB(channels)

    def forward(self, fea_left, fea_right, cost):
        fea_left, fea_right, cost = self.pab1(fea_left, fea_right, cost)
        fea_left, fea_right, cost = self.pab2(fea_left, fea_right, cost)
        fea_left, fea_right, cost = self.pab3(fea_left, fea_right, cost)
        fea_left, fea_right, cost = self.pab4(fea_left, fea_right, cost)

        return fea_left, fea_right, cost


class CascadedPAM(nn.Module):
    def __init__(self, channels):
        super(CascadedPAM, self).__init__()
        self.stage1 = PAM_stage(channels[0])
        self.stage2 = PAM_stage(channels[1])
        self.stage3 = PAM_stage(channels[2])

        # bottleneck in stage 2
        self.b2 = nn.Sequential(
            nn.Conv2d(128 + 96, 96, 1, 1, 0, bias=True),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.1, inplace=True)
        )
        # bottleneck in stage 3
        self.b3 = nn.Sequential(
            nn.Conv2d(96 + 64, 64, 1, 1, 0, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
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

        fea_left, fea_right, cost_s1 = self.stage1(fea_left_s1, fea_right_s1, cost_s0)

        # stage 2: 1/8
        fea_left = F.interpolate(fea_left, scale_factor=2, mode='bilinear')
        fea_right = F.interpolate(fea_right, scale_factor=2, mode='bilinear')
        fea_left = self.b2(torch.cat((fea_left, fea_left_s2), 1))
        fea_right = self.b2(torch.cat((fea_right, fea_right_s2), 1))

        cost_s1_up = [
            F.interpolate(cost_s1[0].view(b, 1, h_s1, w_s1, w_s1), scale_factor=2, mode='trilinear').squeeze(1),
            F.interpolate(cost_s1[1].view(b, 1, h_s1, w_s1, w_s1), scale_factor=2, mode='trilinear').squeeze(1)
        ]

        fea_left, fea_right, cost_s2 = self.stage2(fea_left, fea_right, cost_s1_up)

        # stage 3: 1/4
        fea_left = F.interpolate(fea_left, scale_factor=2, mode='bilinear')
        fea_right = F.interpolate(fea_right, scale_factor=2, mode='bilinear')
        fea_left = self.b3(torch.cat((fea_left, fea_left_s3), 1))
        fea_right = self.b3(torch.cat((fea_right, fea_right_s3), 1))

        cost_s2_up = [
            F.interpolate(cost_s2[0].view(b, 1, h_s2, w_s2, w_s2), scale_factor=2, mode='trilinear').squeeze(1),
            F.interpolate(cost_s2[1].view(b, 1, h_s2, w_s2, w_s2), scale_factor=2, mode='trilinear').squeeze(1)
        ]

        fea_left, fea_right, cost_s3 = self.stage3(fea_left, fea_right, cost_s2_up)

        return [cost_s1, cost_s2, cost_s3]


def morphologic_process(mask):
    b, _, _, _ = mask.shape
    mask = ~mask
    mask_np = mask.cpu().numpy().astype(bool)
    mask_np = morphology.remove_small_objects(mask_np, 20, 2)
    mask_np = morphology.remove_small_holes(mask_np, 10, 2)
    for idx in range(b):
        mask_np[idx, 0, :, :] = morphology.binary_closing(mask_np[idx, 0, :, :], morphology.disk(3))
    mask_np = 1 - mask_np
    mask_np = mask_np.astype(float)

    return torch.from_numpy(mask_np).float().to(mask.device)


def regress_disp(att, valid_mask):
    """
    :param att:         B * H * W * W
    :param valid_mask:  B * 1 * H * W
    """
    b, h, w, _ = att.shape
    index = torch.arange(w).view(1, 1, 1, w).to(att.device).float()  # index: 1*1*1*w
    disp_ini = index - torch.sum(att * index, dim=-1).view(b, 1, h, w)

    # partial conv
    filter1 = torch.zeros(1, 3).to(att.device)
    filter1[0, 0] = 1
    filter1[0, 1] = 1
    filter1 = filter1.view(1, 1, 1, 3)

    filter2 = torch.zeros(1, 3).to(att.device)
    filter2[0, 1] = 1
    filter2[0, 2] = 1
    filter2 = filter2.view(1, 1, 1, 3)

    valid_mask_0 = valid_mask
    disp = disp_ini * valid_mask_0

    valid_mask_num = 1
    while valid_mask_num > 0:
        valid_mask_1 = F.conv2d(valid_mask_0, filter1, padding=[0, 1])
        disp = disp * valid_mask_0 + \
            F.conv2d(disp, filter1, padding=[0, 1]) / (valid_mask_1 + 1e-4) * (
                    (valid_mask_1 > 0).float() - valid_mask_0)
        valid_mask_num = (valid_mask_1 > 0).float().sum() - valid_mask_0.sum()
        valid_mask_0 = (valid_mask_1 > 0).float()

    valid_mask_num = 1
    while valid_mask_num > 0:
        valid_mask_1 = F.conv2d(valid_mask_0, filter2, padding=[0, 1])
        disp = disp * valid_mask_0 + \
            F.conv2d(disp, filter2, padding=[0, 1]) / (valid_mask_1 + 1e-4) * (
                    (valid_mask_1 > 0).float() - valid_mask_0)
        valid_mask_num = (valid_mask_1 > 0).float().sum() - valid_mask_0.sum()
        valid_mask_0 = (valid_mask_1 > 0).float()

    return disp_ini * valid_mask + disp * (1 - valid_mask)


class Output(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, cost, max_disp):
        cost_right2left, cost_left2right = cost
        b, h, w, _ = cost_right2left.shape

        # M_right2left
        # exclude negative disparities & disparities larger than max_disp (if available)
        cost_right2left = torch.tril(cost_right2left)
        if max_disp > 0:
            cost_right2left = cost_right2left - torch.tril(cost_right2left, -max_disp)
        cost_right2left = torch.exp(cost_right2left - cost_right2left.max(-1)[0].unsqueeze(-1))
        cost_right2left = torch.tril(cost_right2left)
        if max_disp > 0:
            cost_right2left = cost_right2left - torch.tril(cost_right2left, -max_disp)
        att_right2left = cost_right2left / (cost_right2left.sum(-1, keepdim=True) + 1e-8)

        # M_left2right
        # exclude negative disparities & disparities larger than max_disp (if available)
        cost_left2right = torch.triu(cost_left2right)
        if max_disp > 0:
            cost_left2right = cost_left2right - torch.triu(cost_left2right, max_disp)
        cost_left2right = torch.exp(cost_left2right - cost_left2right.max(-1)[0].unsqueeze(-1))
        cost_left2right = torch.triu(cost_left2right)
        if max_disp > 0:
            cost_left2right = cost_left2right - torch.triu(cost_left2right, max_disp)
        att_left2right = cost_left2right / (cost_left2right.sum(-1, keepdim=True) + 1e-8)

        # valid mask (left image)
        valid_mask_left = torch.sum(att_left2right.detach(), -2) > 0.1
        valid_mask_left = valid_mask_left.view(b, 1, h, w)
        valid_mask_left = morphologic_process(valid_mask_left)

        # disparity
        disp = regress_disp(att_right2left, valid_mask_left)

        # valid mask (right image)
        valid_mask_right = torch.sum(att_right2left.detach(), -2) > 0.1
        valid_mask_right = valid_mask_right.view(b, 1, h, w)
        valid_mask_right = morphologic_process(valid_mask_right)

        # cycle-attention maps
        att_left2right2left = torch.matmul(att_right2left, att_left2right).view(b, h, w, w)
        att_right2left2right = torch.matmul(att_left2right, att_right2left).view(b, h, w, w)

        return disp, \
            (att_right2left.view(b, h, w, w), att_left2right.view(b, h, w, w)), \
            (att_left2right2left, att_right2left2right), \
            (valid_mask_left, valid_mask_right)


class B(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.input = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))

        self.body = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))

    def forward(self, x):
        x = self.input(x)

        return x + self.body(x)


class Upsample(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))

    def forward(self, x):
        return self.upsample(x)


class ColorCorrection(pl.LightningModule):
    def __init__(self, channels):
        super().__init__()

        self.E5_upsample = Upsample(2 * channels[5], 2 * channels[4])
        self.D5 = B(4 * channels[4], 2 * channels[4])
        self.D5_upsample = Upsample(2 * channels[4], 2 * channels[3])
        self.D4 = B(4 * channels[3], 2 * channels[3])
        self.D4_upsample = Upsample(2 * channels[3], 2 * channels[2])
        self.D3 = B(4 * channels[2], 2 * channels[2])
        self.D3_upsample = Upsample(2 * channels[2], 2 * channels[1])
        self.D2 = B(4 * channels[1], 2 * channels[1])
        self.D2_upsample = Upsample(2 * channels[1], 2 * channels[0])
        self.D1 = B(4 * channels[0], 2 * channels[0])

        self.output = B(2 * channels[0], 3)

    def forward(self, left, warped_right):
        fea_identity, fea_E0, fea_E1, fea_E2, fea_E3, fea_E4 = [
            torch.cat([fea_left, fea_right], dim=1) for fea_left, fea_right in zip(left, warped_right)]

        fea_D5 = self.D5(torch.cat((self.E5_upsample(fea_E4), fea_E3), dim=1))
        fea_D4 = self.D4(torch.cat((self.D5_upsample(fea_D5), fea_E2), dim=1))
        fea_D3 = self.D3(torch.cat((self.D4_upsample(fea_D4), fea_E1), dim=1))
        fea_D2 = self.D2(torch.cat((self.D3_upsample(fea_D3), fea_E0), dim=1))
        fea_D1 = self.D1(torch.cat((self.D2_upsample(fea_D2), fea_identity), dim=1))

        return self.output(fea_D1)
