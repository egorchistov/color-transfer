import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class B(nn.Module):
    def __init__(self, in_channels, out_channels, bn):
        super().__init__()

        self.input = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not bn),
            nn.BatchNorm2d(out_channels) if bn else nn.Identity(),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))

        self.body = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=not bn),
            nn.BatchNorm2d(out_channels) if bn else nn.Identity(),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=not bn),
            nn.BatchNorm2d(out_channels) if bn else nn.Identity(),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))

    def forward(self, x):
        x = self.input(x)

        return x + self.body(x)


class Upsample(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bn):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=not bn),
            nn.BatchNorm2d(out_channels) if bn else nn.Identity(),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))

    def forward(self, x):
        return self.upsample(x)


class Downsample(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bn):
        super().__init__()

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=not bn),
            nn.BatchNorm2d(out_channels) if bn else nn.Identity(),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))

    def forward(self, x):
        return self.downsample(x)


class Encoder(nn.Module):
    def __init__(self, channels, bn):
        super().__init__()

        self.encoder = nn.Sequential(
            B(channels[0], channels[1], bn=bn),
            Downsample(channels[1], channels[2], bn=bn),
            B(channels[2], channels[2], bn=bn),
            Downsample(channels[2], channels[3], bn=bn),
            B(channels[3], channels[3], bn=bn),
            Downsample(channels[3], channels[4], bn=bn),
            B(channels[4], channels[4], bn=bn))

    def forward(self, x):
        return self.encoder(x)


class PAB(nn.Module):
    def __init__(self, channels, bn):
        super().__init__()

        self.query = B(channels, channels, bn=bn)
        self.key = B(channels, channels, bn=bn)

    def forward(self, fea_left, fea_right):
        c = fea_left.shape[1]

        query = self.query(fea_left).permute(0, 2, 3, 1).contiguous()  # B * H * W * C
        key = self.key(fea_right).permute(0, 2, 1, 3).contiguous()  # B * H * C * W
        cost_right2left = torch.matmul(query, key) / c  # scale the matching cost

        query = self.query(fea_right).permute(0, 2, 3, 1).contiguous()  # B * H * W * C
        key = self.key(fea_left).permute(0, 2, 1, 3).contiguous()  # B * H * C * W
        cost_left2right = torch.matmul(query, key) / c  # scale the matching cost

        return [cost_right2left, cost_left2right]


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


def output(cost):
    cost_right2left, cost_left2right = cost
    b, h, w, _ = cost_right2left.shape

    # M_right2left
    # exclude negative disparities
    cost_right2left = torch.tril(cost_right2left)
    cost_right2left = torch.exp(cost_right2left - cost_right2left.max(-1)[0].unsqueeze(-1))
    cost_right2left = torch.tril(cost_right2left)
    att_right2left = cost_right2left / (cost_right2left.sum(-1, keepdim=True) + 1e-8)

    # M_left2right
    # exclude negative disparities
    cost_left2right = torch.triu(cost_left2right)
    cost_left2right = torch.exp(cost_left2right - cost_left2right.max(-1)[0].unsqueeze(-1))
    cost_left2right = torch.triu(cost_left2right)
    att_left2right = cost_left2right / (cost_left2right.sum(-1, keepdim=True) + 1e-8)

    # valid mask (left image)
    valid_mask_left = torch.sum(att_left2right.detach(), -2) > 0.1
    valid_mask_left = valid_mask_left.view(b, 1, h, w).float()

    # disparity
    disp = regress_disp(att_right2left, valid_mask_left)

    # valid mask (right image)
    valid_mask_right = torch.sum(att_right2left.detach(), -2) > 0.1
    valid_mask_right = valid_mask_right.view(b, 1, h, w).float()

    # cycle-attention maps
    att_left2right2left = torch.matmul(att_right2left, att_left2right).view(b, h, w, w)
    att_right2left2right = torch.matmul(att_left2right, att_right2left).view(b, h, w, w)

    return disp, \
        (att_right2left.view(b, h, w, w), att_left2right.view(b, h, w, w)), \
        (att_left2right2left, att_right2left2right), \
        (valid_mask_left, valid_mask_right)


class Decoder(nn.Module):
    def __init__(self, channels, bn):
        super().__init__()

        self.decoder = nn.Sequential(
            B(channels[0], channels[1], bn=bn),
            Upsample(channels[1], channels[2], bn=bn),
            B(channels[2], channels[2], bn=bn),
            Upsample(channels[2], channels[3], bn=bn),
            B(channels[3], channels[3], bn=bn),
            Upsample(channels[3], channels[4], bn=bn),
            B(channels[4], channels[5], bn=bn))

    def forward(self, x):
        return self.decoder(x)


class Hourglass(pl.LightningModule):
    def __init__(self, channels, bn):
        super().__init__()

        self.E0 = B(channels[0], channels[1], bn=bn)
        self.E0_downsample = Downsample(channels[1], channels[2], bn=bn)
        self.E1 = B(channels[2], channels[2], bn=bn)
        self.E1_downsample = Downsample(channels[2], channels[3], bn=bn)
        self.E2 = B(channels[3], channels[3], bn=bn)
        self.E2_downsample = Downsample(channels[3], channels[4], bn=bn)

        self.E3 = B(channels[4], channels[4], bn=bn)

        self.E3_upsample = Upsample(channels[4], channels[3], bn=bn)
        self.D2 = B(2 * channels[3], channels[3], bn=bn)
        self.D2_upsample = Upsample(channels[3], channels[2], bn=bn)
        self.D1 = B(2 * channels[2], channels[2], bn=bn)
        self.D1_upsample = Upsample(channels[2], channels[1], bn=bn)
        self.D0 = B(2 * channels[1], channels[5], bn=bn)

    def forward(self, x):
        fea_E0 = self.E0(x)
        fea_E1 = self.E1(self.E0_downsample(fea_E0))
        fea_E2 = self.E2(self.E1_downsample(fea_E1))
        fea_E3 = self.E3(self.E2_downsample(fea_E2))

        fea_D2 = self.D2(torch.cat((self.E3_upsample(fea_E3), fea_E2), dim=1))
        fea_D1 = self.D1(torch.cat((self.D2_upsample(fea_D2), fea_E1), dim=1))
        fea_D0 = self.D0(torch.cat((self.D1_upsample(fea_D1), fea_E0), dim=1))

        return fea_D0
