import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import morphology


class ResB(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False))

    def forward(self, x):
        out = self.body(x)
        return self.lrelu(out + x)


class Encoder(nn.Module):
    def __init__(self, n_blocks, channels_in, channels_out):
        super().__init__()
        body = [nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
        )]
        for i in range(n_blocks):
            body.append(
                ResB(channels_out)
            )
        self.body = nn.Sequential(*body)

    def forward(self, x):
        return self.body(x)


# Parallax-Attention Block
class PAB(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.query = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=True),
        )
        self.key = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=True),
        )

    def forward(self, x_left, x_right, cost):
        """
        x_left : (B * C * H * W)
            features from the left image
        x_right : (B * C * H * W)
            features from the right image
        cost : (B * H * W * W)
            input matching cost
        """
        b, c, h, w = x_left.shape
        fea_left = self.head(x_left)
        fea_right = self.head(x_right)

        # C_right2left
        Q = self.query(fea_left).permute(0, 2, 3, 1).contiguous()  # B * H * W * C
        K = self.key(fea_right).permute(0, 2, 1, 3).contiguous()  # B * H * C * W
        cost_right2left = torch.matmul(Q, K) / c  # scale the matching cost
        cost_right2left = cost_right2left + cost[0]

        # C_left2right
        Q = self.query(fea_right).permute(0, 2, 3, 1).contiguous()  # B * H * W * C
        K = self.key(fea_left).permute(0, 2, 1, 3).contiguous()  # B * H * C * W
        cost_left2right = torch.matmul(Q, K) / c  # scale the matching cost
        cost_left2right = cost_left2right + cost[1]

        return x_left + fea_left, \
            x_right + fea_right, \
            (cost_right2left, cost_left2right)


class PAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pab1 = PAB(channels)
        self.pab2 = PAB(channels)
        self.pab3 = PAB(channels)
        self.pab4 = PAB(channels)

    def forward(self, fea_left, fea_right):
        b, _, h, w = fea_left.shape

        cost = [
            torch.zeros(b, h, w, w).to(fea_right.device),
            torch.zeros(b, h, w, w).to(fea_right.device)
        ]

        fea_left, fea_right, cost = self.pab1(fea_left, fea_right, cost)
        fea_left, fea_right, cost = self.pab2(fea_left, fea_right, cost)
        fea_left, fea_right, cost = self.pab3(fea_left, fea_right, cost)
        fea_left, fea_right, cost = self.pab4(fea_left, fea_right, cost)

        return cost


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


class PASM(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.pam = PAM(channels)
        self.output = Output()

    def forward(self, fea_left, fea_right, max_disp=0):
        cost = self.pam(fea_left, fea_right)

        disp, att, att_cycle, valid_mask = self.output(cost, max_disp)

        return disp, att, att_cycle, valid_mask
