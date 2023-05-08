import torch
import torchvision
import torch.nn.functional as F


def loss_pam_photometric(img_left, img_right, att, valid_mask):
    scale = img_left.shape[2] // valid_mask[0].shape[2]

    att_right2left, att_left2right = att
    valid_mask_left, valid_mask_right = valid_mask

    img_left_scale = F.interpolate(img_left, scale_factor=1 / scale, mode="bilinear", align_corners=False)
    img_right_scale = F.interpolate(img_right, scale_factor=1 / scale, mode="bilinear", align_corners=False)

    img_right_warp = torch.matmul(att_right2left, img_right_scale.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    img_left_warp = torch.matmul(att_left2right, img_left_scale.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    loss = F.l1_loss(img_left_scale * valid_mask_left, img_right_warp * valid_mask_left) + \
        F.l1_loss(img_right_scale * valid_mask_right, img_left_warp * valid_mask_right)

    return loss


def loss_pam_photometric_multiscale(img_left, img_right, att, valid_mask):
    weight = [0.05, 0.15, 0.3, 0.5]
    loss = torch.zeros(1).to(img_left.device)

    for idx_scale in range(len(att)):
        loss_scale = loss_pam_photometric(img_left, img_right, att[idx_scale], valid_mask[idx_scale])

        loss = loss + weight[idx_scale] * loss_scale

    return loss


def loss_pam_cycle(att_cycle, valid_mask):
    b, c, h, w = valid_mask[0].shape
    I = torch.eye(w, w).repeat(b, h, 1, 1).to(att_cycle[0].device)

    att_left2right2left, att_right2left2right = att_cycle
    valid_mask_left, valid_mask_right = valid_mask

    loss = F.l1_loss(att_left2right2left * valid_mask_left.permute(0, 2, 3, 1),
                     I * valid_mask_left.permute(0, 2, 3, 1)) + \
        F.l1_loss(att_right2left2right * valid_mask_right.permute(0, 2, 3, 1),
                  I * valid_mask_right.permute(0, 2, 3, 1))

    return loss


def loss_pam_cycle_multiscale(att_cycle, valid_mask):
    weight = [0.05, 0.15, 0.3, 0.5]
    loss = torch.zeros(1).to(att_cycle[0][0].device)

    for idx_scale in range(len(att_cycle)):
        loss_scale = loss_pam_cycle(att_cycle[idx_scale], valid_mask[idx_scale])

        loss = loss + weight[idx_scale] * loss_scale

    return loss


def loss_pam_smoothness(att):
    att_right2left, att_left2right = att

    loss = F.l1_loss(att_right2left[:, :-1, :, :], att_right2left[:, 1:, :, :]) + \
        F.l1_loss(att_left2right[:, :-1, :, :], att_left2right[:, 1:, :, :]) + \
        F.l1_loss(att_right2left[:, :, :-1, :-1], att_right2left[:, :, 1:, 1:]) + \
        F.l1_loss(att_left2right[:, :, :-1, :-1], att_left2right[:, :, 1:, 1:])

    return loss


def loss_pam_smoothness_multiscale(att):
    weight = [0.05, 0.15, 0.3, 0.5]
    loss = torch.zeros(1).to(att[0][0].device)

    for idx_scale in range(len(att)):
        loss_scale = loss_pam_smoothness(att[idx_scale])

        loss = loss + weight[idx_scale] * loss_scale

    return loss


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

        features = torchvision.models.vgg16(weights="DEFAULT").features

        self.blocks = torch.nn.Sequential(
            features[:4],
            features[4:9],
            features[9:16],
            features[16:23]
        )

        self.blocks.eval()

        for block in self.blocks:
            for p in block.parameters():
                p.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))

    def forward(self, corrected, content_target, style_target):
        corrected = (corrected - self.mean) / self.std
        style_target = (style_target - self.mean) / self.std
        content_target = (content_target - self.mean) / self.std

        loss = 0

        for block in self.blocks:
            corrected = block(corrected)
            style_target = block(style_target)
            content_target = block(content_target)

            loss += F.mse_loss(corrected, content_target)

            _, c, h, w = corrected.shape

            act_x = corrected.reshape(corrected.shape[0], corrected.shape[1], -1)
            act_y = style_target.reshape(style_target.shape[0], style_target.shape[1], -1)
            gram_x = act_x @ act_x.permute(0, 2, 1) / (c * h * w)
            gram_y = act_y @ act_y.permute(0, 2, 1) / (c * h * w)

            loss += 5 * F.mse_loss(gram_x, gram_y)

        return loss
