"""Implementation of the improved color-image-difference metric (iCID
metric) which predicts the perceived difference of two color images.

This implementation is direct translation from matlab source code:
https://github.com/Netflix/vmaf/blob/master/matlab/cid_icid/iCID/iCID_Metric/iCID.m

Different with that implementation spatial prefiltering is not implemented.

Citation
--------
@article{preiss2014color,
  title={Color-image quality assessment: From prediction to optimization},
  author={Preiss, Jens and Fernandes, Felipe and Urban, Philipp},
  journal={IEEE Transactions on Image Processing},
  volume={23},
  number={3},
  pages={1366--1378},
  year={2014},
  publisher={IEEE}
}
"""

import torch
from kornia.color import rgb_to_lab
from torchvision.transforms.functional import gaussian_blur


def icid(
    img1,
    img2,
    intent="perceptual",
    omit_maps67=False,
    downsampling=True,
):
    # Non-variable parameters
    alpha = 3
    kernel_size = [11, 11]
    sigma = [2.0, 2.0]

    # Process parameters
    if intent == "perceptual":
        weights = torch.tensor([0.002, 10, 10, 0.002, 0.002, 10, 10])
    elif intent == "hue-preserving":
        weights = torch.tensor([0.002, 10, 10, 0.002, 0.02, 10, 10])
    elif intent == "chromatic":
        weights = torch.tensor([0.002, 10, 10, 0.02, 0.02, 10, 10])
    else:
        raise ValueError("Intent should be either 'perceptual', 'hue-preserving', or 'chromatic'")

    if omit_maps67:
        exponents = torch.tensor([1, 1, alpha, 1, 1, 0, 0])
    else:
        exponents = torch.tensor([1, 1, alpha, 1, 1, 1, 1])

    # TRANSFORM IMAGES TO THE WORKING COLOR SPACE
    # Here we use the almost perceptually uniform and hue linear LAB color space.

    # Downsample images
    if downsampling:
        height, width = img1.shape[-2:]
        f = max(1, round(min(height, width) / 256))
        if f > 1:
            img1 = torch.nn.functional.interpolate(img1, scale_factor=1 / f, mode="bilinear")
            img2 = torch.nn.functional.interpolate(img2, scale_factor=1 / f, mode="bilinear")

    # Transform images
    img1 = rgb_to_lab(img1)
    img2 = rgb_to_lab(img2)

    # CALCULATE PREMAPS
    # Calculating the premaps is based upon the SSIM from MeTriX MuX

    # Abbreviations
    L1 = img1[..., 0, :, :]
    A1 = img1[..., 1, :, :]
    B1 = img1[..., 2, :, :]
    C1_sq = A1 ** 2 + B1 ** 2
    C1 = torch.sqrt(C1_sq)

    L2 = img2[..., 0, :, :]
    A2 = img2[..., 1, :, :]
    B2 = img2[..., 2, :, :]
    C2_sq = A2 ** 2 + B2 ** 2
    C2 = torch.sqrt(C2_sq)

    # Mean intensity mu
    muL1 = gaussian_blur(L1, kernel_size, sigma)
    muC1 = gaussian_blur(C1, kernel_size, sigma)
    muL2 = gaussian_blur(L2, kernel_size, sigma)
    muC2 = gaussian_blur(C2, kernel_size, sigma)

    # Standard deviation sigma
    sL1_sq = gaussian_blur(L1 ** 2, kernel_size, sigma) - muL1 ** 2
    sL1_sq[sL1_sq < 0] = 0
    sL1 = torch.sqrt(sL1_sq)
    sL2_sq = gaussian_blur(L2 ** 2, kernel_size, sigma) - muL2 ** 2
    sL2_sq[sL2_sq < 0] = 0
    sL2 = torch.sqrt(sL2_sq)

    sC1_sq = gaussian_blur(C1 ** 2, kernel_size, sigma) - muC1 ** 2
    sC1_sq[sC1_sq < 0] = 0
    sC1 = torch.sqrt(sC1_sq)
    sC2_sq = gaussian_blur(C2 ** 2, kernel_size, sigma) - muC2 ** 2
    sC2_sq[sC2_sq < 0] = 0
    sC2 = torch.sqrt(sC2_sq)

    # Get mixed terms (dL_sq, dC_sq, dH_sq, sL12)
    dL_sq = (muL1 - muL2) ** 2
    dC_sq = (muC1 - muC2) ** 2
    H = (A1 - A2) ** 2 + (B1 - B2) ** 2 - (C1 - C2) ** 2
    H[H < 0] = 0
    dH_sq = gaussian_blur(torch.sqrt(H), kernel_size, sigma) ** 2
    sL12 = gaussian_blur(L1 * L2, kernel_size, sigma) - muL1 * muL2
    sC12 = gaussian_blur(C1 * C2, kernel_size, sigma) - muC1 * muC2

    # CALCULATE MAPS
    maps_inv = torch.stack([
        # 1) Lightness difference
        1 / (weights[0] * dL_sq + 1),

        # 2) Lightness contrast
        (weights[1] + 2 * sL1 * sL2) / (weights[1] + sL1_sq + sL2_sq),

        # 3) Lightness structure
        (weights[2] + abs(sL12)) / (weights[2] + sL1 * sL2),

        # 4) Chroma difference
        1 / (weights[3] * dC_sq + 1),

        # 5) Hue difference
        1 / (weights[4] * dH_sq + 1),

        # 6) Chroma contrast
        (weights[5] + 2 * sC1 * sC2) / (weights[5] + sC1 ** 2 + sC2 ** 2),

        # 7) Chroma structure
        (weights[6] + torch.abs(sC12)) / (weights[6] + sC1 * sC2)
    ], dim=1)

    # CALCULATE PREDICTION
    # Potentiate maps with exponents
    maps_inv **= exponents[None, :, None, None]

    # Compute prediction pixel wise
    prediction = 1 - torch.mean(torch.prod(maps_inv, dim=1))

    # Occasionally, the prediction has a very small imaginary part; we keep
    # only the real part of the prediction
    prediction = prediction.real

    return prediction
